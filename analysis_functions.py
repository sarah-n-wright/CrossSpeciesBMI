import pandas as pd
import numpy as np
import networkx as nx
import ndex2
import getpass

## Utilities ------------------------------------------------------------

def num_to_mp(number):
    mp = "MP:"
    num = str(number)
    zeros_to_add = 7-len(num)
    mp = [mp] + ["0"] * zeros_to_add + [num]
    return "".join(mp)


def get_MP_description(term, MPO):
    return MPO.node_attr.loc[term].description


def _get_mp_graph(datafile="parsed_mp.txt"):
    mp_data = pd.read_csv(datafile, sep="\t", header=None)
    mp_data.head()
    mp_graph = nx.from_pandas_edgelist(mp_data, 0,1, create_using=nx.DiGraph)
    return mp_graph


def get_top_level_terms(mp_graph, root="MP:0000001" ,exclude=["MP:0003012", "MP:0002873"]):
    return [node for node in nx.dfs_preorder_nodes(mp_graph, root, 1) if node not in exclude][1:]


def change_symbols(mgi_data, pc_node_map):
    symbol_map = pd.Series(pc_node_map.index.values, index=pc_node_map["symbol"]).to_dict()
    mgi_data["human_ortholog"] = mgi_data["human_ortholog"].map(symbol_map)
    return mgi_data


def get_gene_hits_no_annotation(genes, term, MPO, term_mapping):
    term_genes = [MPO.genes[idx] for idx in term_mapping[term]]
    overlap = set(genes).intersection(set(term_genes))
    return overlap


def load_pcnet():
    interactome_uuid='4de852d9-9908-11e9-bcaf-0ac135e8bacf' # for PCNet
    # interactome_uuid='275bd84e-3d18-11e8-a935-0ac135e8bacf' # for STRING high confidence
    ndex_server='public.ndexbio.org'
    ndex_user=None
    ndex_password=None
    G_int = ndex2.create_nice_cx_from_server(
            ndex_server, 
            username=ndex_user, 
            password=ndex_password, 
            uuid=interactome_uuid
        ).to_networkx()
    nodes = list(G_int.nodes)

    # pcnet appears to have some self edges... should remove them. 
    G_int.remove_edges_from(nx.selfloop_edges(G_int))

    # print out interactome num nodes and edges for diagnostic purposes
    print('number of nodes:')
    print(len(G_int.nodes))
    print('\nnumber of edges:')
    print(len(G_int.edges))
    return nodes, G_int


def load_BMI_network():
    ndex_server='public.ndexbio.org'
    ndex_user=None
    ndex_password=None
    G_overlap_cx = ndex2.create_nice_cx_from_server(
            ndex_server, 
            username=ndex_user, 
            password=ndex_password, 
            uuid='e8cc9239-d91a-11eb-b666-0ac135e8bacf')
    G_overlap = G_overlap_cx.to_networkx()
    print('number of nodes:')
    print(len(G_overlap.nodes))
    print('\nnumber of edges:')
    print(len(G_overlap.edges))
    return G_overlap


## Enrichment Analysis ---------------------------------------------------

def genes_per_node(MPO):
    node_order = MPO.topological_sorting(top_down=False)
    nodes = [i for i in node_order]
    results = {i: set(MPO.term_2_gene[i]) for i in node_order}
    genes = {i: set(MPO.gene_2_term[i]) for i in MPO.genes}
    while len(nodes) > 0:
        current = nodes.pop()
        children = MPO.parent_2_child[current]
        if len(children) > 0:
            for child in children:
                if child != current:
                    results[current] = results[current].union(results[child])
        for gene in results[current]:
            if gene not in genes.keys():
                genes[gene] = set([current])
            else:
                genes[gene] = genes[gene].union(set([current]))
        else:
            pass
    counts = {k: len(results[k]) for k in results.keys()}
    return counts, genes, results


def community_term_enrichment(community_name, hier_df, MPO, mgi_df, term_counts, gene_to_terms, keep_genes=None, exclude_genes=None):
    """
    :param community_name:
    :param hier_df:
    :param MPO: Mammalian phenotype ontology with gene info
    :param mgi_df: Gene phenotype mapping
    :param term_counts: Number of genes associated with each term
    :param gene_to_terms: Number of terms associated with each gene
    :param keep_genes: List of genes to maintain in the analysis - ONLY these genes will be kept
    :param exclude_genes: List of genes to exclude from the analysis
    """
    # get the genes in the community
    genes = hier_df.loc[community_name, "CD_MemberList"]
    if type(genes) is str:  # split into a list of genes
        genes_all = genes.split(" ")
        N_hier = len(genes_all)
    # only keep genes in the MGI ontology    
    genes = [ g for g in genes_all if g in MPO.genes ]  
    
    # subset genes based on input
    if keep_genes is not None:
        genes = [g for g in genes if g in keep_genes]
        N_hier = len([g for g in genes_all if g in keep_genes])
    if exclude_genes is not None:
        genes = [g for g in genes if g not in exclude_genes]
        N_hier = len([g for g in genes_all if g not in exclude_genes])
    
    # exit if there are no genes remaining
    if len(genes) == 0:
        print("0/"+str(len(genes_all)), "in MPO.genes/seeds")
        return pd.DataFrame()
    
    # Get the terms associated with these genes
    terms = []
    for gene in genes:
        terms += list(gene_to_terms[MPO.genes.index(gene)])

    # Join term totals and observed
    to_test = pd.DataFrame(pd.Series(terms, name="observed").value_counts()).join(pd.Series(term_counts, name="total"))
    M_pool_size = len(G_int.nodes())
    
    # Get odss ratio, p value of odds ratio, and 95% confidence interval
    OR_test = to_test.apply(lambda x: get_contingency_stats(x.observed, x.total, N_hier, M_pool_size), axis=1)
    try: 
        OR_test = pd.concat(list(OR_test), ignore_index=True)
    except TypeError:
        print(OR_test)
        print(N_hier, terms)
    OR_test.index = to_test.index
    to_test = pd.concat([to_test, OR_test], axis=1)
    
    to_test = to_test.assign(hyper_p=lambda z: hypergeom.sf(k=z.observed, M=M_pool_size, n=z.total, N=N_hier))
    desc = MPO.node_attr.loc[to_test.index]
    to_test = to_test.assign(sig_5e6=to_test["hyper_p"] < 5e-6)
    to_test = to_test.join(desc, how="left")
    to_test = to_test.assign(size=N_hier)
    return to_test


def get_contingency_stats(observed, term_size, community_size, network_size):
    q00 = observed
    q01 = term_size - observed
    q10 = community_size - observed
    q11 = network_size - q00 - q01 - q10
    results_table = [[q00, q01], [q10, q11]]
    #print(results_table)
    CT = contingency_tables.Table2x2(results_table)
    OR_p_temp = CT.oddsratio_pvalue()
    OR_CI_temp = CT.oddsratio_confint()
    OR = CT.oddsratio
    #print(CT.chi2_contribs)
    #return CT
    return pd.DataFrame({"OR":OR, "OR_p": OR_p_temp, "OR_CI_lower":OR_CI_temp[0], "OR_CI_upper":OR_CI_temp[1]}, index=[0])

## Results ------------------------------------------------------------------------------

def get_hits(network_results, data, p=0.01, OR=2, obs_min=3, total=10000, level= 3):
    mp_graph = _get_mp_graph()
    node_levels = nx.shortest_path_length(mp_graph, "MP:0000001")
    term_totals = data.loc[:, ("total", "description")]
    network_results = network_results.join(term_totals, how="left").drop_duplicates(subset=["OR", "total"])
    network_results = network_results.assign(depth=[node_levels[node] for node in network_results.index])
    try:
        hits = network_results.loc[network_results.obs >= obs_min].loc[network_results.OR >= OR]
        hits = hits.loc[network_results.q >= -1 * np.log10(p)]
        hits = hits.loc[network_results.total <= total]
        hits = hits.loc[network_results.depth >= level]
        # if there are parents in the hits then keep the parents and not the children
        subG = G.subgraph(hits.index)       
        keep_nodes = [node for node in subG.nodes if subG.in_degree[node]==0]
        #keep_nodes = [node for node in subG.nodes if (subG.in_degree[node] == 0 and subG.out_degree[node] == 0) or 
        #             (subG.in_degree[node] == 0 and subG.out_degree[node] > 1) or 
        #              (subG.out_degree[node] == 0 and subG.out_degree[subG.predecessors(node)[0]] == 1)]
        hits = hits.loc[keep_nodes]
        return hits
    except:
        print("no hits passing filters")