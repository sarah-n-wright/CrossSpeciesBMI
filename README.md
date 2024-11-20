## Introduction
This repository contains code and data used perform the analysis described in 
**Wright, S.N., Leger, B. *et al*. Genome-wide association studies of human and rat BMI converge on synapse, epigenome, and hormone signaling networks**. *Cell Reports, 2023*.

## Requirements
This work was perfomed using `python 3.9.13` and used the following libraries:
* [netcoloc v0.1.6.post1](https://pypi.org/project/netcoloc/0.1.6.post1/) See additional installation instuctions at 
https://github.com/ucsd-ccbb/NetColoc
* [matplotlib-venn v0.11.9](https://pypi.org/project/matplotlib-venn/0.11.9/)  
* [cdapsutil v0.2.0a1](https://pypi.org/project/cdapsutil/0.2.0a1/)
* [ddot for python3](https://github.com/idekerlab/ddot/tree/python3). This is required for `netcoloc` functioning, and can be installed form source:

    ```
    wget https://github.com/idekerlab/ddot/archive/refs/heads/python3.zip
    unzip python3.zip
    cd ddot-python3
    python setup.py bdist_wheel
    pip install dist/ddot*py3*whl
    ```
    or via Github:
    ```
    git clone --branch python3 https://github.com/idekerlab/ddot.git
    cd ddot
    python setup.py bdist_wheel
    pip install dist/ddot*py3*whl
    ```

A full specification of the environment used for code development can be found in `environment.yml`

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
* `Create_Rat_STRING.ipynb` Generation of rat molecular interaction networks from STRING DB. Defines high and mid-confidence networks based on edge scores.

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
  * Gene-level summary statistics for human BMI, human height, rat BMI & rat body length
  * Seed gene lists for 4 negative control traits
  * UKB data for heritability and genetic correlation of control traits
* `Data/outputs/`
  * Network propagation results, including sampling results
  * Conserved systems map hierarchy data
  * Final community annotations
  * Full MGD enrichment results
  * Combined systems map hierarchy data (conserved + human-specific + rat-specific)
  * Tissue enrichment results
  * List of subgraph genes
* `Data/Reference/`
  * Stable versions of Mouse Genome Database data. These files can be updated by 
running `updated_netcoloc_functions.load_MPO()` and 
`updated_netcoloc_functions.load_MGI_mouseKO_data()` with the option `update=True`.
  * Stable version of HCOP rat-human ortholog mapping
  
## Other
* `Figures/` python produced versions of figures used in the manuscript

## Original summary statistics
* `rat BMI, body length, body weight:` https://library.ucsd.edu/dc/object/bb83725195
* `human BMI:` bmi.giant-ukbb.meta-analysis.combined.23May2018.txt.gz @ https://zenodo.org/records/1251813#.XCLJ7vZKhE4
* `human height:` https://portals.broadinstitute.org/collaboration/giant/images/6/63/Meta-analysis_Wood_et_al%2BUKBiobank_2018.txt.gz
