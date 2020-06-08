# PDTB 2.0/3.0 Preprocessing and Evaluation

This repository contains preprocessing and evaluation code used in the paper Implicit Discourse Relation Classification: We Need to Talk about Evaluation (link forthcoming), accepted to ACL 2020.

## Data

Access to both PDTB 2.0 and 3.0 require an LDC license, so we cannot distribute this data. 

PDTB 3.0 can be downloaded here: https://catalog.ldc.upenn.edu/LDC2019T05

Our preprocessing code for PDTB 2.0 requires it to be in `.csv` format. Please refer to [this repository](https://github.com/cgpotts/pdtb2) for obtaining data in csv format.

## Requirements

* Python 3.7.3 or higher

* For preprocessing PDTB 3.0, no packages are required other than numpy.

* For preprocessing PDTB 2.0, please first run:

      bash pdtb2_setup.sh
