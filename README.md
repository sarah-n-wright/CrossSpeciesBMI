## Introduction
This repository contains code and data used perform the analysis described in 
**Wright, S.N., Leger, B. *et al*. Genome-wide association studies of human and rat body mass index converge on a 
conserved molecular network**.

## Requirements
This work was perfomed using `python 3.9.13` and used the following libraries:
* `netcoloc` versions > . See installation instuctions at 
https://github.com/ucsd-ccbb/NetColoc
* `matplotlib-venn`
* `itertools`
* `python-louvain`

## Notebooks
**Main Notebooks:**  
* `1_Rat_Human_BMI_Network_Colocalization.ipynb`: Definition of seed gene sets, 
network colocalization of rat and human BMI, and statistical evaluation of conserved 
BMI network and control traits. 
* `2_Rat_Human_BMI_Systems_Map.ipynb`: Hierarchical clustering of the conserved
BMI network, annotation of the systems map, and validation with mouse genetic 
perturbation data.
* `3_Species_Specific_BMI_Networks.ipynb`: Definition of independent species BMI 
networks and formation of a combined systems hierarchy.   

**Supplemental Notebooks:**  
* `rat_human_bmi_cluster_loci.ipynb`: Analysis of the number of unique genetic loci 
represented by seed genes in each community of the conserved BMI systems map. 

## Function Files
* `analysis_functions.py` Functions not contained or modified from NetColoc 
used to generate the results in this study. 
* `plotting_functions.py` Functions used for plotting figures generated for 
this study.  
* `updated_netcoloc_functions.py` Contains modifications to the published NetColoc 
source code for use in this project. These modifications include:
  * ability to maintain stable versions of Mouse Genome Database data and phenotype 
ontology.
  * implementation of network propagation sampling procedure for gene sets > 500 genes
  * implementation of partitioning the seed genes and non-seed genes for permuted
distributions of NPS.
  * calculation of mean NPS score and permuted distribution of mean NPS score

## Data files
* `Data/inputs/`
  * Seed gene lists
  * Gene-level summary statistics for human BMI, human height, rat BMI & TODO rat body length
* `Data/outputs/`
  * Network propagation results, including sampling results
  * Conserved systems map hierarchy data
  * Final community annotations
  * Full MGD enrichment results
  * Combined systems map hierarchy data (conserved + human-specific + rat-specific)
  * Tissue enrichment results
* `Data/Reference/`
  * Stable versions of Mouse Genome Database data. These files can be updated by 
running `updated_netcoloc_functions.load_MPO()` and 
`updated_netcoloc_functions.load_MGI_mouseKO_data()` with the option `update=True`.
  * Stable version of HCOP rat-human ortholog mapping
  * Stable version of summary rat eQTL data from [ratgtex.org]
  
## Other
* `Figures/` python produced versions of figures used in the manuscript
* `rat_genetics.yaml` provided to define environment used to develop the above code. 
