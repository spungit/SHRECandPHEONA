## Overview
This repository contains all the code to use open-source LLMs to classify concepts and phenotype patients for Acute Respiratory Failure respiratory support treatments.

## Citation
1. Pungitore SA, Yadav S, Subbian V. PHEONA: An Evaluation Framework for Large Language Model-based Approaches to Computational Phenotyping. AMIA Annu Symp Proc. 2026 Feb 14;2025:1041–50. PubMed PMID: 41726409; PubMed Central PMCID: PMC12919548.

2. Pungitore S, Yadav S, Douglas M, Mosier J, Subbian V. SHREC: A framework for advancing next-generation computational phenotyping with large language models. PLOS Digital Health. 2026 Feb 13;5(2):e0001217. doi:10.1371/journal.pdig.0001217

3. Pungitore S, Yadav S, Maughan D, Subbian V. Lightweight Language Models are Prone to Reasoning Errors for Complex Computational Phenotyping Tasks [Internet]. arXiv; 2026. Available from: http://arxiv.org/abs/2507.23146

## Description of Code
Folder Descriptions:

1) Data Processing: Includes processing the data for concept classification and phenotyping.
2) Ground Truths: Includes script required to determine the ground truths from the eICU database. Requires setup of the eICU database using Postgres. (https://eicu-crd.mit.edu/gettingstarted/dbsetup/)
3) PHEONA for Phenotyping: Includes code to run the PHEONA evaluation tests for the phenotyping task specifically.
4) RunClassification: Includes code to run the LLM classification of the constructed concepts.
5) RunPhenotyping: Includes code to run the LLM phenotyping of the constructed descriptions after data has been processed following concept classification.
6) PHEONA Reasoning Assessment: Includes code to perform the reasoning assessment for the additional reasoning component of PHEONA for the task of computational phenotyping.

Run the Ollama server using the following terminal command:

OLLAMA_HOST=127.0.0.1:11434 ollama serve

Run the main.py files using the following terminal command (assuming Python is installed and a virtual environment has been setup):

python3 main.py mistral 11434
