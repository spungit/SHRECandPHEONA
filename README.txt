## Overview
This repository contains all the code to use open-source LLMs to classify concepts and phenotype patients for Acute Respiratory Failure respiratory support treatments.

## Citation


## Description of Code
Folder Descriptions:

1) Data Processing: Includes processing the data for concept classification and phenotyping.
2) Ground Truths: Includes script required to determine the ground truths from the eICU database. Requires setup of the eICU database using Postgres. (https://eicu-crd.mit.edu/gettingstarted/dbsetup/)
3) PHEONA for Phenotyping: Includes code to run the PHEONA evaluation tests for the phenotyping task specifically.
4) RunClassification: Includes code to run the LLM classification of the constructed concepts.
5) RunPhenotyping: Includes code to run the LLM phenotyping of the constructed descriptions after data has been processed following concept classification.