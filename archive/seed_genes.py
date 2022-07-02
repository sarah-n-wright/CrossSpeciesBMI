from json import load


import pandas as pd
import numpy as np
from analysis_functions import load_pcnet

def load_human_seed_genes(filepath, interactome_nodes, trait=''):
    """
    :param filepath:
    :param interactome_nodes:
    :param trait:
    """
    all_scores = pd.read_csv(filepath, sep="\t", index_col='gene_symbol')
    # subset to genes in the interactome
    all_scores = all_scores.loc[list(np.intersect1d(all_scores.index.tolist(), interactome_nodes))]
    # Calculate bonferroni corrected pvalue (alpha=0.05)
    bonf_p = .05/len(all_scores)
    # Get significant genes
    seeds = all_scores[all_scores['pvalue'] < bonf_p].index.tolist()
    print("Number of",trait,"seeds:", len(seeds))
    return seeds

if __name__ == "__main__":
    pc_nodes, G = load_pcnet()
    human_seeds = load_human_seed_genes("Documents/RatGenetics/onNRNB/GIANT_Genomics/BMI/GIANT_BMI_pascal.sum.genescores.txt", pc_nodes)
    pd.DataFrame({"gene":human_seeds}).to_csv("Documents/RatGenetics/human_seed_genes.txt", index=False, header=False)
    human_seeds=set(human_seeds)
    rat_seeds = set(pd.read_csv("Documents/RatGenetics/onNRNB/seed_genes/ratBMI_seed_relaxed.txt")["0"].values)
    overlap = human_seeds.intersection(rat_seeds)
    overlap_df = pd.DataFrame({"gene":list(overlap)})
    overlap_df.to_csv("Documents/RatGenetics/overlapping_seed_genes.txt", index=False, header=False)
    