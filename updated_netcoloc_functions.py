import pandas as pd
import ddot
from ddot import Ontology
import requests
import numpy as np
from os.path import exists
import mygene
from tqdm import tqdm
from scipy.stats import norm
import math
import random as rn
mg = mygene.MyGeneInfo()
from netcoloc.netprop_zscore import calculate_heat_zscores
import matplotlib.pyplot as plt
import seaborn as sns


## Modified NetColoc functions -------------------------------------------------------------------------------
def calculate_expected_overlap(z_scores_1, z_scores_2, seed1=None, seed2=None, 
                                z_score_threshold=3.0, z1_threshold=1.5,
                                z2_threshold=1.5,
                                num_reps=1000, plot=False, overlap_control=None):
    """ Determines size of expected network overlap by randomly
    shuffling gene names
    
    **Modifications:**
    * `seed1`, `seed2`: These inputs are used to provide the input seed gene lists used for network propagation for identification of overlapping seed genes when `overlap_control != None`. 
    * `overlap_control`: Genes that are seeds from both input traits (overlapping seeds) are expected to have higher network colocalization scores by definition. This bias can now be maintained or excluded in the permutation analysis. `overlap_control == "bin"` performs the permutation separately for the group of overlapping seeds and `overlap_control == "remove"` excludes overlapping seeds from the calculation. 

    Args:
        z_scores_1 (:py:class:`pandas.Series`, :py:class:`pandas.DataFrame`): Result from :py:func:`~netcoloc.netprop_zscore.netprop_zscore` or :py:func:`~netcoloc.netprop_zscore.calculate_heat_zscores`
                    containing the z-scores of each gene following network propagation. The index consists of gene names
        z_scores_2 (:py:class:`pandas.Series`, :py:class:`pandas.DataFrame`): Similar to **z_scores_1**. This and **z_scores_1** must contain the same genes (ie. come from the same interactome network)
        seed1 (_type_, optional): The list of seed genes associated with the first trait (used to generate z_scores_1). Required if `overlap_control != None`.. Defaults to None.
        seed2 (_type_, optional): The list of seed genes associated with the second trait (used to generate z_scores_2). Required if `overlap_control != None`.. Defaults to None.
        z_score_threshold (float, optional): threshold to determine whether a gene is a part of the network overlap or not. Genes with combined z-scores
                    below this threshold will be discarded. Defaults to 3.
        z1_threshold (float, optional): individual z1-score threshold to determine whether a gene is a part of the network overlap or not. Genes with z1-scores
                    below this threshold will be discarded. Defaults to 1.5.
        z2_threshold (float, optional): individual z2-score threshold to determine whether a gene is a part of the network overlap or not. Genes with z2-scores
                    below this threshold will be discarded. Defaults to 1.5.
        num_reps (int, optional): The number of permutation steps to perform. Defaults to 1000.
        plot (bool, optional): If ``True``, distribution will be plotted. Defaults to False.
        overlap_control (str, optional): 'bin' to permute overlapping seed genes separately, 'remove' to not consider overlapping seed genes.. Defaults to None.

    Returns:
        float: The size of the colocalized network defined by the input z scores and thresholds
        list: Vector of network sizes from thresholds applied to permuted data
    """
    # Build a distribution of expected network overlap sizes by shuffling node names
    random_network_overlap_sizes = []
    if isinstance(z_scores_1, pd.Series):
        z_scores_1 = pd.DataFrame(z_scores_1, columns=["z"])           
    if isinstance(z_scores_2, pd.Series):
        z_scores_2 = pd.DataFrame(z_scores_2, columns=["z"])
    z1z2 = z_scores_1.join(z_scores_2, lsuffix="1", rsuffix="2")
    z1z2 = z1z2.assign(zz=z1z2.z1 * z1z2.z2)
    if overlap_control == "remove":  # take out any genes that were seed for both traits
        seed_overlap = list(set(seed1).intersection(set(seed2)))
        print("Overlap seed genes:", len(seed_overlap))
        z1z2.drop(seed_overlap, axis=0, inplace=True)
    elif overlap_control == "bin":  # create a separate dataframe for overlapping seeds and then exclude from main data
        seed_overlap = list(set(seed1).intersection(set(seed2)))
        print("Overlap seed genes:", len(seed_overlap))
        overlap_z1z2 = z1z2.loc[seed_overlap]
        overlap_z1 = np.array(overlap_z1z2.z1)
        overlap_z2 = np.array(overlap_z1z2.z2)
        z1z2.drop(seed_overlap, axis=0, inplace=True)
    z1 = np.array(z1z2.z1)
    z2 = np.array(z1z2.z2)
    # Calculate the observed network size
    network_overlap_size = len(calculate_network_overlap(z1z2.z1, z1z2.z2,
                                                        z_score_threshold=z_score_threshold,
                                                        z1_threshold=z1_threshold,
                                                        z2_threshold=z2_threshold))
    if overlap_control == "bin":  # if overlapping seed genes were separated add their contribution
        network_overlap_size += len(calculate_network_overlap(overlap_z1z2.z1, overlap_z1z2.z2,
                                                        z_score_threshold=z_score_threshold,
                                                        z1_threshold=z1_threshold,
                                                        z2_threshold=z2_threshold))
    random_network_overlap_sizes = np.zeros(num_reps)
    # perform the permutations
    for i in tqdm(range(num_reps)):
        perm_z1z2 = np.zeros(len(z1))
        rn.shuffle(z1)
        perm_size = len(calculate_network_overlap(z1, z2,
                                                        z_score_threshold=z_score_threshold,
                                                        z1_threshold=z1_threshold,
                                                        z2_threshold=z2_threshold))
        if overlap_control == "bin":  # perform the separate permutation
            overlap_perm_z1z2 = np.zeros(len(overlap_z1))
            rn.shuffle(overlap_z1) 
            perm_size_overlap = len(calculate_network_overlap(overlap_z1, overlap_z2,
                                                        z_score_threshold=z_score_threshold,
                                                        z1_threshold=z1_threshold,
                                                        z2_threshold=z2_threshold))
            perm_size += perm_size_overlap  # add to the size
        random_network_overlap_sizes[i] = perm_size
    if plot:
        plt.figure(figsize=(5, 4))
        dfig = sns.histplot(random_network_overlap_sizes,
                            label='Expected network intersection size')
        plt.vlines(network_overlap_size, ymin=0, ymax=dfig.dataLim.bounds[3], color='r',
                    label='Observed network intersection size')
        plt.xlabel('Size of proximal subgraph, z > ' + str(z_score_threshold),
                    fontsize=16)
        plt.legend(fontsize=12)

    return network_overlap_size, random_network_overlap_sizes


## validaton ---------------------------------------------------------------------------------

def load_MPO(url='http://www.informatics.jax.org/downloads/reports/MPheno_OBO.ontology', use_genes=False,
            mapping=None, restrict_to=None, data_loc='', update=False):
    """Modified from :py:func:`netcoloc.validation.load_MPO`
    
    Function to parse and load mouse phenotype ontology, using DDOT's ontology module
    
    **Modifications:**
    * `use_genes`: allows mapping of gene-term relationships to the ontology object
    * `mapping`: the gene-term relationship data, required if `use_genes==True`
    * `restrict_to`:
    * `use_display`: option to toggle :py:func:`display` on or off
    * `data_loc`: path to directory where data should be saved to, and where the function should search for existing data 
    if `update==True`
    * `update`: if `update==False` existing data files will not be overwritten, allowing the user to maintain a stable version. If 
    `update==True` then the latest version will be downloaded. 


    Args:
        url (str, optional): URL containing MPO ontology file. Defaults to 'http://www.informatics.jax.org/downloads/reports/MPheno_OBO.ontology'.
        use_genes (bool, optional): _description_. Defaults to False.
        mapping (pd.DataFrame, optional): Gene term mappings to used if `use_genes==True`. Defaults to None.
        restrict_to (list, optional): Specify a subset of genes to be included from the mapping. Defaults to None.
        data_loc (str, optional): Location to save data and search for existing data. Defaults to ''.
        update (bool, optional): Should existing data be replaced with latest data (if data already exists in `data_loc`)?. Defaults to False.

    Raises:
        ImportError: If DDOT package is not found
        AssertionError: If use_genes=True, but no mapping provided.

    Returns:
        :py:class:`ddot.Ontology`: MPO parsed using DDOT
    """
    if use_genes:
        assert mapping is not None, "You must supply a mapping dataframe if use_genes==True"
    # check if there is existing data, and whether it should be updated
    # >>>> begin modification
    if (not exists(data_loc + 'MPheno_OBO.ontology')) or update:
    # <<< end modification
        # download the mammalian phenotype ontology, parse with ddot
        r = requests.get(url, allow_redirects=True)
        open(data_loc + 'MPheno_OBO.ontology', 'wb').write(r.content)
        ddot.parse_obo(data_loc + 'MPheno_OBO.ontology',
                    data_loc + 'parsed_mp.txt',
                    data_loc + 'id2name_mp.txt',
                    data_loc + 'id2namespace_mp.txt',
                    data_loc + 'altID_mp.txt')

    MP2desc = pd.read_csv(data_loc + 'id2name_mp.txt', sep='\t', names=['MP', 'description'], index_col='MP')

    MP2desc = MP2desc.loc[MP2desc.index.dropna()]  # drop NAN from index
    print(len(MP2desc))

    hierarchy = pd.read_table(data_loc + 'parsed_mp.txt',
                                sep='\t',
                                header=None,
                                names=['Parent', 'Child', 'Relation', 'Namespace'])
    # >>> begin modification
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
    else:
    # <<< end modification
        # create the ontology without gene mappings
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
                          map_using="mygeneinfo", update=False, data_loc=""):
    """Modified from :py:func:`netcoloc.validation.load_MGI_mouseKO_data`
    
    Function to parse and load mouse knockout data from MGI.
    
    **Modifications:**
    * `map_using`: allows gene in MGI to be mapped to human orthologs using either the default (mygeneinfo) or the MGI's mapping file
    * `data_loc`: path to directory where data should be saved to, and where the function should search for existing data 
    if `update==True`
    * `update`: if `update==False` existing data files will not be overwritten, allowing the user to maintain a stable version. If 
    `update==True` then the latest version will be downloaded. 

    Args:
        url (str, optional): url location of MGI knockout data. Defaults to 'http://www.informatics.jax.org/downloads/reports/MGI_PhenoGenoMP.rpt'.
        map_using (str, optional): Database to map gene names to human, one of "mygeneinfo" of "mgi". Defaults to "mygeneinfo".
        update (bool, optional): Should existing data be replaced with latest data (if data already exists in `data_loc`)?. Defaults to False.
        data_loc (str, optional): Location to save data and search for existing data. Defaults to "".

    Returns:
        :py:class:`pandas.DataFrame`: parsed MGI knockout dataframe, including column for human orthologs
    """
    # download MGI phenotype data
    if (not exists(data_loc + 'MGI_PhenoGenoMP.rpt')) or update:
        r = requests.get(url, allow_redirects=True)
        open(data_loc + 'MGI_PhenoGenoMP.rpt', 'wb').write(r.content)

    # parse the downloaded MGI phenotype data
    mgi_df = pd.read_csv(data_loc + 'MGI_PhenoGenoMP.rpt', sep='\t',
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
        if not exists(data_loc + 'MRK_List2.rpt') or update:
            keep_url = "http://www.informatics.jax.org/downloads/reports/MRK_List2.rpt"
            r_map = requests.get(keep_url, allow_redirects=True)
            open(data_loc + 'MRK_List2.rpt', 'wb').write(r_map.content)
        keep = pd.read_csv(data_loc + 'MRK_List2.rpt', sep="\t", usecols=["MGI Accession ID", "Marker Symbol",
                            "Feature Type", "Marker Name"])
        keep = keep.loc[keep["Feature Type"].isin(["protein coding gene"])].reset_index(drop=True)
        mgi_df["MGI"] = mgi_df.MGI_marker_accession_ID.apply(lambda x: x.split("|"))
        mgi_df = mgi_df.explode("MGI", ignore_index=True)
        mgi_df["MGI"] = [mg if type(mg) is str else mg[0] for mg in mgi_df.MGI]
        mgi_df = mgi_df.loc[mgi_df["MGI"].isin(keep["MGI Accession ID"])]
        mgi_df = mgi_df.merge(keep.loc[:, ("MGI Accession ID", "Marker Symbol")], left_on="MGI",
                            right_on="MGI Accession ID", how="left")

        if not exists(data_loc + 'HMD_HumanPhenotype.rpt') or update:
            map_url = "http://www.informatics.jax.org/downloads/reports/HMD_HumanPhenotype.rpt"
            r_map = requests.get(map_url, allow_redirects=True)
            open(data_loc + 'HMD_HumanPhenotype.rpt', 'wb').write(r_map.content)
        mapping = pd.read_csv(data_loc + 'HMD_HumanPhenotype.rpt', sep="\t", header=None, usecols=[0, 2, 3],
                            index_col=False, names=["symbol", "gene_name", "MGI"])
        mapping = mapping.loc[mapping["MGI"].isin(keep["MGI Accession ID"])]

        mg_mapped = mgi_df.merge(mapping, on="MGI", how="left")
        mg_mapped.loc[mg_mapped.symbol.isna(), "gene_name"] = mg_mapped.loc[mg_mapped.symbol.isna(), "Marker Symbol"]
        mg_mapped = mg_mapped.drop_duplicates()
        mg_mapped.rename(columns={"symbol": 'human_ortholog'}, inplace=True)
        return mg_mapped
    else:
        print("Please specify a mapping database, either 'mygeneinfo' or 'mgi'. Raw data returned.")
        return mgi_df


def calculate_network_overlap(z_scores_1, z_scores_2, z_score_threshold=3,
                              z1_threshold=1.5, z2_threshold=1.5):
    """Modified from :py:func:`netcoloc.network_colocalization.calculate_network_overlap`
    
    Function to determine which genes overlap. Returns a list of the
    overlapping genes
    
    **Modifications:**
    * Now accepts :py:class:`pandas.Series`, :py:class:`pandas.DataFrame`, or :py:class:`numpy.ndarray` as inputs for `z_scores_1`
    and `z_scores_2`

    Args:
        z_scores_1 (:py:class:`pandas.Series`, :py:class:`pandas.DataFrame`, :py:class:`numpy.ndarray`):Result from :py:func:`~netcoloc.netprop_zscore.netprop_zscore`
                    or :py:func:`~netcoloc.netprop_zscore.calculate_heat_zscores` containing the z-scores of each gene following network propagation. The index consists of gene names.
        z_scores_2 (:py:class:`pandas.Series`, :py:class:`pandas.DataFrame`, :py:class:`numpy.ndarray`): Similar to **z_scores_1**. This and **z_scores_1**
                    must contain the same genes (ie. come from the same interactome network)
        z_score_threshold (int, optional): threshold to determine whether a gene is a part of the network overlap or not. Genes with combined z-scores
                    below this threshold will be discarded. Defaults to 3.
        z1_threshold (float, optional): individual z1-score threshold to determine whether a gene is a part of the network overlap or not. Genes with z1-scores
                    below this threshold will be discarded. Defaults to 1.5.
        z2_threshold (float, optional): individual z2-score threshold to determine whether a gene is a part of the network overlap or not. Genes with z2-scores
                    below this threshold will be discarded. Defaults to 1.5.. Defaults to 1.5.

    Returns:
        list: genes in the network overlap (genes with high combined z-scores)
    """
    if isinstance(z_scores_1, pd.Series):
        z_scores_1 = z_scores_1.to_frame(name='z_scores_1')
        z_scores_2 = z_scores_2.to_frame(name='z_scores_2')
    elif isinstance(z_scores_1, np.ndarray):
        z_scores_1 = pd.DataFrame({"z_scores_1":z_scores_1})
        z_scores_2 = pd.DataFrame({"z_scores_2":z_scores_2})
    else:
        z_scores_1.columns = ["z_scores_1"]
        z_scores_2.columns = ["z_scores_2"]
    z_scores_joined = z_scores_1.join(z_scores_2)
    z_scores_combined = (z_scores_joined['z_scores_1']
                        * z_scores_joined['z_scores_2']
                        * (z_scores_joined['z_scores_1'] > 0)
                        * (z_scores_joined['z_scores_2'] > 0))
    # get rid of unlikely genes which have low scores in either z1 or z2
    high_z_score_genes = z_scores_combined[
        (z_scores_combined >= z_score_threshold)
        & (z_scores_joined['z_scores_1'] > z1_threshold)
        & (z_scores_joined['z_scores_2'] > z2_threshold)
    ].index.tolist()

    return high_z_score_genes


## Extensions to NetColoc ------------------------------------------------------------------------------------
def get_p_from_permutation_results(observed, permuted):
    """Calculates the significance of the observed mean relative to the empirical normal distribution of permuted means.

    Args:
        observed (float): The observed value to be tested
        permuted (list): List of values that make up the expected distribution
    
    Returns:
        float: p-value from z-test of observed value versus the permuted distribution
    """
    p = norm.sf((observed-np.mean(permuted))/np.std(permuted))
    p = round(p, 4 - int(math.floor(math.log10(abs(p)))) - 1)
    return p


def calculate_mean_z_score_distribution(z1, z2, num_reps=1000, zero_double_negatives=True, 
                                        overlap_control="remove", seed1=[], seed2=[]):
    """Determines size of expected mean combined `z=z1*z2` by randomly shuffling gene names

    Args:
        z1 (pd.Series, pd.DataFrame): Vector of z-scores from network propagation of trait 1
        z2 (pd.Series, pd.DataFrame): Vector of z-scores from network propagation of trait 2
        num_reps (int): Number of perumation analyses to perform. Defaults to 1000
        zero_double_negatives (bool, optional): Should genes that have a negative score in both `z1` and `z2` be ignored? Defaults to True.
        overlap_control (str, optional): 'bin' to permute overlapping seed genes separately, 'remove' to not consider overlapping seed genes. Any other value will do nothing. Defaults to "remove".
        seed1 (list, optional): List of seed genes used to generate `z1`. Required if `overlap_control!=None`. Defaults to [].
        seed2 (list, optional): List of seed genes used to generate `z2`. Required if `overlap_control!=None`. Defaults to [].

    Returns:
        float: The observed mean combined z-score from network colocalization
        list: List of permuted mean combined z-scores
    """
    if isinstance(z1, pd.Series):
        z1 = pd.DataFrame(z1, columns=["z"])
    if isinstance(z2, pd.Series):
        z2 = pd.DataFrame(z2, columns=["z"])
    z1z2 = z1.join(z2, lsuffix="1", rsuffix="2")
    z1z2 = z1z2.assign(zz=z1z2.z1 * z1z2.z2)
    #print(z1z2.head())
    if overlap_control == "remove":
        seed_overlap = list(set(seed1).intersection(set(seed2)))
        print("Overlap seed genes:", len(seed_overlap))
        z1z2.drop(seed_overlap, axis=0, inplace=True)
    elif overlap_control == "bin":
        seed_overlap = list(set(seed1).intersection(set(seed2)))
        print("Overlap seed genes:", len(seed_overlap))
        overlap_z1z2 = z1z2.loc[seed_overlap]
        overlap_z1 = np.array(overlap_z1z2.z1)
        z1z2.drop(seed_overlap, axis=0, inplace=True)
    z1 = np.array(z1z2.z1)
    z2 = np.array(z1z2.z2)
    if zero_double_negatives:
        for node in z1z2.index:
            if (z1z2.loc[node].z1 < 0 and z1z2.loc[node].z2 < 0):
                z1z2.loc[node, 'zz'] = 0
    permutation_means = np.zeros(num_reps)
    for i in tqdm(range(num_reps)):
        perm_z1z2 = np.zeros(len(z1))
        np.random.shuffle(z1)

        for node in range(len(z1)):
            if not zero_double_negatives or not (z1[node] < 0 and z2[node] < 0):
                perm_z1z2[node] = z1[node] * z2[node]
            else:
                perm_z1z2[node] = 0
        if overlap_control == "bin":
            overlap_perm_z1z2 = np.zeros(len(overlap_z1))
            np.random.shuffle(overlap_z1) 
            for node in range(len(overlap_z1)):
                if zero_double_negatives and (overlap_z1[node] < 0 and z2[node] < 0):
                    overlap_perm_z1z2[node] = 0
                else:
                    overlap_perm_z1z2[node] = overlap_z1[node] * z2[node]
            perm_z1z2 = np.concatenate([perm_z1z2, overlap_perm_z1z2])
                    
        permutation_means[i] = np.mean(perm_z1z2)
    return np.mean(z1z2.zz), permutation_means


def calculate_heat_zscores_with_sampling(data, nodes, individual_heats, G_PC, trait="BMI", max_genes=500, num_samples=100,
                                        nominal_sig=0.05, num_reps=1000, out_path="", minimum_bin_size=10):
    """Takes a set of summary statistics and a molecular interaction and performs sampling of the significant genes.
    For each sample a random selection of seed genes is chosen, weighted by the p-value of each gene in the summary
    statistics. Network propagation with zscore calculation is performed for each sample to generate a distribution
    of z-scores for each gene in the seed_gene set.

    Args:
        data (pd.DataFrame): Gene level summary statistics
        nodes (list): list of nodes in the interaction network
        individual_heats (np.array): Heat matrix calculated by `netprop_zscore.get_individual_heats_matrix()`
        G_PC (nx.Graph): molecular interaction network
        trait (str, optional): name of trait being investigated. Defaults to "BMI".
        max_genes (int, optional): Maximum number of seed genes to include in each sample (maximum=500). Defaults to 500.
        num_samples (int, optional): Number of times to perform sampling. Defaults to 100.
        nominal_sig (float, optional): Significance cutoff for keeping genes in data (Note: this value will be Bonferroni corrected). Defaults to 0.05.
        num_reps (int, optional): Number of repetitions of randomization for generating null distribution for z_scores. Defaults to 1000.
        out_path (str, optional): File path prefix for saving results of sampling. Defaults to "".
        minimum_bin_size (int, optional): minimum number of genes that should be in each degree matching bin. Defaults to 10.

    Returns:
        pd.DataFrame: Gene x sampling run dataframe of sampled z-scores
    """
    assert max_genes <= 500, "NetColoc is only valid for sets of 500 or less genes so maximum number of genes for sampling must be <= 500"
    outfile = out_path + trait + "sampling_" + str(max_genes) + "_" + str(num_samples) + ".tsv"
    data = data.loc[data.gene_symbol.isin(nodes)]  # subset to genes in interaction network
    all_seeds = data.loc[data.pvalue <= nominal_sig / len(data)]  # Bonferroni correction
    all_seeds = all_seeds.assign(log10p=-1 * np.log10(all_seeds.pvalue))  # get -log10p for weighted sampling
    sampling_results = []
    for i in range(num_samples):
        # perform propagation for sample
        sample_seeds = rn.choices(population=all_seeds.gene_symbol.values, weights=all_seeds.log10p.values, k=max_genes)
        sample_results = calculate_heat_zscores(individual_heats, nodes=list(G_PC.nodes), degrees=dict(G_PC.degree),
                                                seed_genes=sample_seeds, num_reps=num_reps,
                                                minimum_bin_size=minimum_bin_size, random_seed=i)[0]
        sample_z = pd.DataFrame(sample_results, columns=["z" + str(i)])
        # save running results of sampling
        if i == 0:
            sample_z.to_csv(outfile, sep="\t")
        else:
            existing = pd.read_csv(outfile, sep="\t", index_col=0)
            existing = existing.join(sample_z)
            existing.to_csv(outfile, sep="\t")
        sampling_results.append(sample_z)
    return pd.concat(sampling_results, axis=1)
