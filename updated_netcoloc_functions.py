import pandas as pd
import ddot
from ddot import Ontology
import requests
import numpy as np
from os.path import exists
import mygene
mg = mygene.MyGeneInfo()


def load_MPO(url='http://www.informatics.jax.org/downloads/reports/MPheno_OBO.ontology', use_genes=False,
             mapping=None, restrict_to=None, use_display=False):
    """
    Function to parse and load mouse phenotype ontology, using DDOT's ontology module

    :param url: URL containing MPO ontology file
    :type url: str
    :param use_genes:
    :type use_genes: bool
    :param mapping:
    :type mapping:
    :param restrict_to:
    :type restrict_to:
    :param use_display:
    :type use_display: bool
    :return: MPO parsed using DDOT
    :rtype: :py:class:`ddot.Ontology`
    :raises ImportError: If DDOT package is not found
    """
    if use_genes:
        assert mapping is not None, "You must supply a mapping dataframe if use_genes==True"
    # download the mammalian phenotype ontology, parse with ddot
    r = requests.get(url, allow_redirects=True)
    open('MPheno_OBO.ontology', 'wb').write(r.content)
    ddot.parse_obo('MPheno_OBO.ontology',
                   'parsed_mp.txt',
                   'id2name_mp.txt',
                   'id2namespace_mp.txt',
                   'altID_mp.txt')

    MP2desc = pd.read_csv('id2name_mp.txt', sep='\t', names=['MP', 'description'], index_col='MP')

    MP2desc = MP2desc.loc[MP2desc.index.dropna()]  # drop NAN from index
    print(len(MP2desc))

    hierarchy = pd.read_table('parsed_mp.txt',
                              sep='\t',
                              header=None,
                              names=['Parent', 'Child', 'Relation', 'Namespace'])

    if use_genes:  # map genes from `mapping` to terms in the ontology
        mouse_mapping = mapping.loc[:, ("human_ortholog", "MP")].dropna().reset_index()
        mouse_mapping = mouse_mapping.loc[:, ("human_ortholog", "MP")]
        mouse_mapping.columns = ["Gene", "Term"]
        if restrict_to is not None:  # restrict to specified subset of genes
            mouse_mapping = mouse_mapping.loc[mouse_mapping.Gene.isin(restrict_to)]
        # generate ontology using gene mapping
        MPO = Ontology.from_table(
            table=hierarchy,
            parent='Parent',
            child='Child',
            add_root_name='MP:00SUPER',
            ignore_orphan_terms=True,
            mapping=mouse_mapping,
            mapping_parent='Term',
            mapping_child='Gene')
    else:  # create the ontology without gene mappings
        MPO = Ontology.from_table(
            table=hierarchy,
            parent='Parent',
            child='Child',
            add_root_name='MP:00SUPER',
            ignore_orphan_terms=True)

    # add description to node attribute
    terms_keep = list(np.unique(hierarchy['Parent'].tolist()+hierarchy['Child'].tolist()))
    MPO.node_attr = MP2desc.loc[terms_keep]

    return MPO


def load_MGI_mouseKO_data(url='http://www.informatics.jax.org/downloads/reports/MGI_PhenoGenoMP.rpt',
                          map_using="mygeneinfo", update=False):
    """
    Function to parse and load mouse knockout data from MGI.

    :param url: location of MGI knockout data
    :type url: str
    :return: parsed MGI knockout dataframe, including column for human orthologs
    :rtype: :py:class:`pandas.DataFrame`
    """
    # download MGI phenotype data
    if (not exists('MGI_PhenoGenoMP.rpt')) or update:
        r = requests.get(url, allow_redirects=True)
        open('MGI_PhenoGenoMP.rpt', 'wb').write(r.content)

    # parse the downloaded MGI phenotype data
    mgi_df = pd.read_csv('MGI_PhenoGenoMP.rpt', sep='\t',
                         names=['MGI_Allele_Accession_ID',
                                'Allele symbol', 'involves',
                                'MP', 'PMID', 'MGI_marker_accession_ID'])
    if map_using == "mygeneinfo":
        # extract gene names
        gene_name = [a.split('<')[0] for a in mgi_df['Allele symbol'].tolist()]
        mgi_df['gene_name'] = gene_name
        mgi_df.index = mgi_df['gene_name']
        # map mouse genes to human orthologs
        mouse_genes = list(np.unique(mgi_df['gene_name']))
        mg_mapped = mg.querymany(mouse_genes,
                                 as_dataframe=True, species=['mouse', 'human'],
                                 scopes='symbol', fields='symbol')
        # drop genes with no human ortholog
        print(len(mg_mapped))
        mg_mapped = mg_mapped.dropna(subset=['symbol'])
        print(len(mg_mapped))
        # drop duplicates
        mg_mapped = mg_mapped[~mg_mapped.index.duplicated(keep='first')]
        print(len(mg_mapped))
        mgi_df['human_ortholog'] = mgi_df['gene_name'].map(dict(mg_mapped['symbol']))
        return mgi_df

    elif map_using == "mgi":
        if not exists('MRK_List2.rpt') or update:
            keep_url = "http://www.informatics.jax.org/downloads/reports/MRK_List2.rpt"
            r_map = requests.get(keep_url, allow_redirects=True)
            open('MRK_List2.rpt', 'wb').write(r_map.content)
        keep = pd.read_csv('MRK_List2.rpt', sep="\t", usecols=["MGI Accession ID", "Marker Symbol",
                                                                     "Feature Type", "Marker Name"])
        keep = keep.loc[keep["Feature Type"].isin(["protein coding gene"])].reset_index(drop=True)
        mgi_df["MGI"] = mgi_df.MGI_marker_accession_ID.apply(lambda x: x.split("|"))
        mgi_df = mgi_df.explode("MGI", ignore_index=True)
        mgi_df["MGI"] = [mg if type(mg) is str else mg[0] for mg in mgi_df.MGI]
        mgi_df = mgi_df.loc[mgi_df["MGI"].isin(keep["MGI Accession ID"])]
        mgi_df = mgi_df.merge(keep.loc[:, ("MGI Accession ID", "Marker Symbol")], left_on="MGI",
                              right_on="MGI Accession ID", how="left")

        if not exists('HMD_HumanPhenotype.rpt') or update:
            map_url = "http://www.informatics.jax.org/downloads/reports/HMD_HumanPhenotype.rpt"
            r_map = requests.get(map_url, allow_redirects=True)
            open('HMD_HumanPhenotype.rpt', 'wb').write(r_map.content)
        mapping = pd.read_csv('HMD_HumanPhenotype.rpt', sep="\t", header=None, usecols=[0, 2, 3],
                              index_col=False, names=["symbol", "gene_name", "MGI"])
        mapping = mapping.loc[mapping["MGI"].isin(keep["MGI Accession ID"])]

        mg_mapped = mgi_df.merge(mapping, on="MGI", how="left")
        mg_mapped.loc[mg_mapped.symbol.isna(), "gene_name"] = mg_mapped.loc[mg_mapped.symbol.isna(), "Marker Symbol"]
        mg_mapped = mg_mapped.drop_duplicates()
        mg_mapped.rename(columns={"symbol": 'human_ortholog'}, inplace=True)
        return mg_mapped
