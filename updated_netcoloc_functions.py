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


## Modified NetColoc functions -------------------------------------------------------------------------------
## Added to netcoloc
def calculate_expected_overlap(z_scores_1, z_scores_2, seed1=None, seed2=None, 
                               z_score_threshold=3, z1_threshold=1.5,
                               z2_threshold=1.5,
                               num_reps=1000, plot=False, overlap_control=None):
    """
    Determines size of expected network overlap by randomly
    shuffling gene names

    :param z_scores_1: Result from :py:func:`~netcoloc.netprop_zscore.netprop_zscore`
                       or :py:func:`~netcoloc.netprop_zscore.calculate_heat_zscores`
                       containing the z-scores of each gene following network
                       propagation. The index consists of gene names
    :type z_scores_1: :py:class:`pandas.Series`
    :param z_scores_2: Similar to **z_scores_1**. This and **z_scores_1**
                       must contain the same genes (ie. come from the same
                       interactome network)
    :type z_scores_2: :py:class:`pandas.Series`
    :param z_score_threshold: threshold to determine whether a gene is
        a part of the network overlap or not. Genes with combined z-scores
        below this threshold will be discarded
    :type z_score_threshold: float
    :param z1_threshold: individual z1-score threshold to determine whether a gene is
        a part of the network overlap or not. Genes with z1-scores
        below this threshold will be discarded
    :type z1_threshold: float
    :param z2_threshold: individual z2-score threshold to determine whether a gene is
        a part of the network overlap or not. Genes with z2-scores
        below this threshold will be discarded
    :type z2_threshold: float
    :param num_reps:
    :param plot: If ``True``, distribution will be plotted
    :type plot: bool
    :return:
    :rtype: float
    """
    # Build a distribution of expected network overlap sizes by shuffling node names
    random_network_overlap_sizes = []
    if isinstance(z_scores_1, pd.Series):
        z_scores_1 = pd.DataFrame(z_scores_1, columns=["z"])           
    if isinstance(z_scores_2, pd.Series):
        z_scores_2 = pd.DataFrame(z_scores_2, columns=["z"])
    z1z2 = z_scores_1.join(z_scores_2, lsuffix="1", rsuffix="2")
    z1z2 = z1z2.assign(zz=z1z2.z1 * z1z2.z2)
    if overlap_control == "remove":
        seed_overlap = list(set(seed1).intersection(set(seed2)))
        print("Overlap seed genes:", len(seed_overlap))
        z1z2.drop(seed_overlap, axis=0, inplace=True)
    elif overlap_control == "bin":
        seed_overlap = list(set(seed1).intersection(set(seed2)))
        print("Overlap seed genes:", len(seed_overlap))
        overlap_z1z2 = z1z2.loc[seed_overlap]
        overlap_z1 = np.array(overlap_z1z2.z1)
        overlap_z2 = np.array(overlap_z1z2.z2)
        z1z2.drop(seed_overlap, axis=0, inplace=True)
    z1 = np.array(z1z2.z1)
    z2 = np.array(z1z2.z2)
    network_overlap_size = len(calculate_network_overlap(z1z2.z1, z1z2.z2,
                                                         z_score_threshold=z_score_threshold,
                                                         z1_threshold=z1_threshold,
                                                         z2_threshold=z2_threshold))
    network_overlap_size += len(calculate_network_overlap(overlap_z1z2.z1, overlap_z1z2.z2,
                                                         z_score_threshold=z_score_threshold,
                                                         z1_threshold=z1_threshold,
                                                         z2_threshold=z2_threshold))
    random_network_overlap_sizes = np.zeros(num_reps)
    for i in tqdm(range(num_reps)):
        perm_z1z2 = np.zeros(len(z1))
        rn.shuffle(z1)
        perm_size = len(calculate_network_overlap(z1, z2,
                                                         z_score_threshold=z_score_threshold,
                                                         z1_threshold=z1_threshold,
                                                         z2_threshold=z2_threshold))
        if overlap_control == "bin":
            overlap_perm_z1z2 = np.zeros(len(overlap_z1))
            rn.shuffle(overlap_z1) 
            perm_size_overlap = len(calculate_network_overlap(overlap_z1, overlap_z2,
                                                         z_score_threshold=z_score_threshold,
                                                         z1_threshold=z1_threshold,
                                                         z2_threshold=z2_threshold))
            
            perm_size += perm_size_overlap
                    
        
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
    """
    Modified from :py:func:`netcoloc.validation.load_MPO`
    
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

    :param url: URL containing MPO ontology file
    :type url: str
    :param use_genes: Should gene-term mappings be included in the ontology object? default=False
    :type use_genes: bool
    :param mapping: Gene term mappings to used if `use_genes==True`
    :type mapping: :py:class:`pandas.DataFrame`
    :param restrict_to: Specify a subset of genes to be included from the mapping
    :type restrict_to: list
    :param data_loc: Location to save data and search for existing data
    :type data_loc: str
    :param update: Should existing data be replaced with latest data (if data already exists in `data_loc`)? default=False
    :type update: bool
    
    :return: MPO parsed using DDOT
    :rtype: :py:class:`ddot.Ontology`
    :raises ImportError: If DDOT package is not found
    :raises AssertionError: If use_genes=True, but no mapping provided.
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
    """
    Modified from :py:func:`netcoloc.validation.load_MGI_mouseKO_data`
    
    Function to parse and load mouse knockout data from MGI.
    
    **Modifications:**
    * `map_using`: allows gene in MGI to be mapped to human orthologs using either the default (mygeneinfo) or the MGI's mapping file
    * `data_loc`: path to directory where data should be saved to, and where the function should search for existing data 
    if `update==True`
    * `update`: if `update==False` existing data files will not be overwritten, allowing the user to maintain a stable version. If 
    `update==True` then the latest version will be downloaded. 

    :param url: location of MGI knockout data
    :type url: str
    :param map_using: Database to map gene names to human, one of "mygeneinfo" of "mgi"
    :type map_using: string
    :param data_loc: Location to save data and search for existing data
    :type data_loc: str
    :param update: Should existing data be replaced with latest data (if data already exists in `data_loc`)? default=False
    :type update: bool
       
    :return: parsed MGI knockout dataframe, including column for human orthologs
    :rtype: :py:class:`pandas.DataFrame`
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
    """
    Modified from :py:func:`netcoloc.network_colocalization.calculate_network_overlap`
    
    Function to determine which genes overlap. Returns a list of the
    overlapping genes
    
    **Modifications:**
    * Now accepts :py:class:`pandas.Series`, :py:class:`pandas.DataFrame`, or :py:class:`numpy.ndarray` as inputs for `z_scores_1`
    and `z_scores_2`

    :param z_scores_1: Result from :py:func:`~netcoloc.netprop_zscore.netprop_zscore`
                       or :py:func:`~netcoloc.netprop_zscore.calculate_heat_zscores`
                       containing the z-scores of each gene following network
                       propagation. The index consists of gene names
    :type z_scores_1: :py:class:`pandas.Series`, :py:class:`pandas.DataFrame`, :py:class:`numpy.ndarray`
    :param z_scores_2: Similar to **z_scores_1**. This and **z_scores_1**
                       must contain the same genes (ie. come from the same
                       interactome network)
    :type z_scores_2: :py:class:`pandas.Series`, :py:class:`pandas.DataFrame`, :py:class:`numpy.ndarray`
    :param z_score_threshold: threshold to determine whether a gene is
        a part of the network overlap or not. Genes with combined z-scores
        below this threshold will be discarded
    :type z_score_threshold: float
    :param z1_threshold: individual z1-score threshold to determine whether a gene is
        a part of the network overlap or not. Genes with z1-scores
        below this threshold will be discarded
    :type z1_threshold: float
    :param z2_threshold: individual z2-score threshold to determine whether a gene is
        a part of the network overlap or not. Genes with z2-scores
        below this threshold will be discarded
    :type z2_threshold: float
    :return: genes in the network overlap (genes with high combined
            z-scores)
    :rtype: list
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

def calculate_significance_colocalized_network_DELETE(z_dict, seed_dict, num_permutations=100, verbose=True, zthresh=3,
                                              overlap_control=None, d1=None, d2=None):
    """
    :param z_dict:
    :type z_dict:
    :param seed_dict:
    :type seed_dict:
    :param num_permutations:
    :param verbose:
    :param zthresh:
    :param overlap_control:
    :param d1:
    :param d2:
    """

    focal_diseases = list(z_dict.keys())
    focal_diseases.reverse()
    # get all combinations of diseases
    focal_combinations = itertools.combinations(focal_diseases, 2)
    # set up data frame to store results
    network_overlap = pd.DataFrame({"trait_pair": focal_combinations})
    network_overlap = network_overlap.reindex(columns=["num_overlap", "obs_exp", "pval_overlap", 
                                                       "exp_mean_overlap", "exp_std_overlap"], 
                                              index = network_overlap.trait_pair.values)
    # iterate over trait pairs
    for trait_pair in network_overlap.index.values:
        #extract the two traits
        d1, d2 = trait_pair
        # get the associated seed genes
        seed1 = seed_dict[d1]
        seed2 = seed_dict[d2]
        # get the z scores following network propagation
        z1=z_dict[d1]
        z2=z_dict[d2] 
        if overlap_control == "remove":
            overlapping_seeds = np.intersect1d(seed1,seed2)
            z1_noseeds = z1.drop(overlapping_seeds, axis=0, inplace=True)
            z2_noseeds = z2.drop(overlapping_seeds, axis=0, inplace=True)
        elif overlap_control == "bin":
            overlapping_seeds = np.intersect1d(seed1,seed2)
            overlap_z1 = z1.loc[overlapping_seeds]
            overlap_z2 = z2.loc[overlapping_seeds]
            z1_noseeds = z1.drop(overlapping_seeds, axis=0, inplace=True)
            z2_noseeds = z2.drop(overlapping_seeds, axis=0, inplace=True)

        if preserve_topology == False:   
            # perform permutations to calculate expected size of colocalized network
            if overlap_control in ["remove", "bin"]:
                z_d1d2_size,high_z_rand=network_colocalization.calculate_expected_overlap(z1_noseeds['z'],z2_noseeds['z'],
                                                                                      plot=False,num_reps=num_permutations,
                                                                                      z_score_threshold=zthresh,
                                                                                      z1_threshold=1.0,
                                                                                      z2_threshold=1.0)
                if overlap_control == "bin":
                    z_d1d2_size_overlap, high_z_rand_overlap = network_colocalization.calculate_expected_overlap(overlap_z1['z'],overlap_z2['z'],
                                                                                      plot=False,num_reps=num_permutations,
                                                                                      z_score_threshold=zthresh,
                                                                                      z1_threshold=1.0,
                                                                                      z2_threshold=1.0)
                    z_d1d2_size += z_d1d2_size_overlap
                    high_z_rand = list(np.array(high_z_rand) + np.array(high_z_rand_overlap))
            else:
                z_d1d2_size,high_z_rand=network_colocalization.calculate_expected_overlap(z1['z'],z2['z'],
                                                                                      plot=False,num_reps=num_permutations,
                                                                                      z_score_threshold=zthresh,
                                                                                      z1_threshold=1.0,
                                                                                      z2_threshold=1.0)
        # store statistics in results table
        ztemp = (z_d1d2_size-np.mean(high_z_rand))/np.std(high_z_rand)
        ptemp = norm.sf(ztemp)
        obs_exp_temp = float(z_d1d2_size)/np.mean(high_z_rand)
    
        network_overlap.loc[[trait_pair]] = [z_d1d2_size,obs_exp_temp, ptemp,np.mean(high_z_rand), np.std(high_z_rand)]
        if verbose:
            print(d1+' + '+d2+'\ninclude seeds')
            print('overlapping seeds = '+str(len(np.intersect1d(seed1,seed2))))
            print("Observed overlap:", z_d1d2_size)
            print("Observed/expected:",obs_exp_temp)
            print("Probability of observed", ptemp)
            print('\n')   
        
    return(network_overlap)



## Added to netcoloc
def get_p_from_permutation_results(observed, permuted):
    """
    Calculates the significance of the observed mean relative to the empirical normal distribution of permuted means.
    :param observed: observed mean NPS_hr
    :type observed: float
    :param permuted: vector of means from permuted NPS_hr
    :type permuted: list
    :return: 
    :rtype: float
    """
    p = norm.sf((observed-np.mean(permuted))/np.std(permuted))
    p = round(p, 4 - int(math.floor(math.log10(abs(p)))) - 1)
    return(p)

def calculate_mean_z_score_distribution(z1, z2, num_reps, zero_double_negatives=True, 
                               overlap_control="remove", seed1=[], seed2=[]):
    """
    :param z1:
    :param z2:
    :param num_reps:
    :param zero_double_negatives:
    :param remove_seeds:
    :param seed1:
    :param seed2:
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
    
def calculate_heat_zscores_with_sampling(data, nodes, individual_heats, G_PC, trait="BMI", max_genes=500,
                                         num_samples=100,
                                         nominal_sig=0.05, num_reps=1000, out_path="", minimum_bin_size=10):
    """
    Takes a set of summary statistics and a molecular interaction and performs sampling of the significant genes.
    For each sample a random selection of seed genes is chosen, weighted by the p-value of each gene in the summary
    statistics. Network propagation with zscore calculation is performed for each sample to generate a distribution
    of z-scores for each gene in the seed_gene set.
    :param data: Gene level summary statistics
    :param nodes: list of nodes in the interaction network
    :param individual_heats: Heat matrix calculated by `netprop_zscore.get_individual_heats_matrix())
    :param G_PC: molecular interaction network
    :param trait: name of trait being investigated
    :param max_genes: Maximum number of seed genes to include in each sample (default=500, maximum=500)
    :param num_samples: Number of times to perform sampling (default=100)
    :param nominal_sig: Significance cutoff for keeping genes in data (Note: this value will be Bonferroni corrected)
    :param num_reps: Number of repetitions of randomization for generating null distribution for z_scores
    :param out_path: prefix for saving results of sampling
    :param minimum_bin_size: minimum number of genes that should be in
            each degree matching bin
    :type minimum_bin_size: int
    :return:
    """
    assert max_genes <= 500, "NetColoc is only valid for sets of 500 or less genes so maximum number of genes for " \
                             "sampling must be <= 500"
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

