# PDTB 2.0/3.0 Preprocessing and BERT/XLNet Baselines

This repository contains the preprocessing code and code to replicate the experiments used in the paper [Implicit Discourse Relation Classification: We Need to Talk about Evaluation](https://www.aclweb.org/anthology/2020.acl-main.480/), accepted to ACL 2020.

Please cite our paper using the following BibTeX entry if you use our code:
```
@inproceedings{kim-etal-2020-implicit,
    title = "Implicit Discourse Relation Classification: We Need to Talk about Evaluation",
    author = "Kim, Najoung  and
      Feng, Song  and
      Gunasekara, Chulaka  and
      Lastras, Luis",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.480",
    doi = "10.18653/v1/2020.acl-main.480",
    pages = "5404--5414",
    abstract = "Implicit relation classification on Penn Discourse TreeBank (PDTB) 2.0 is a common benchmark task for evaluating the understanding of discourse relations. However, the lack of consistency in preprocessing and evaluation poses challenges to fair comparison of results in the literature. In this work, we highlight these inconsistencies and propose an improved evaluation protocol. Paired with this protocol, we report strong baseline results from pretrained sentence encoders, which set the new state-of-the-art for PDTB 2.0. Furthermore, this work is the first to explore fine-grained relation classification on PDTB 3.0. We expect our work to serve as a point of comparison for future work, and also as an initiative to discuss models of larger context and possible data augmentations for downstream transferability.",
}
```

## Data

Access to both PDTB 2.0 and 3.0 requires an LDC license, so we cannot distribute this data. 

* PDTB 3.0 can be downloaded here: https://catalog.ldc.upenn.edu/LDC2019T05

* Our preprocessing code for PDTB 2.0 requires it to be in `.csv` format. Please refer to [this repository](https://github.com/cgpotts/pdtb2) for obtaining data in csv format.

## Requirements

* Python 3.7.3 or higher (Python 3.6 will throw a `UnicodeDecodeError`)

* For preprocessing PDTB 3.0, no packages are required other than numpy.

* For preprocessing PDTB 2.0, you additionally need NLTK. Also please first run:

      bash pdtb2_setup.sh

## Preprocessing

Our preprocessing code can generate section-wise cross-validation folds as described in our paper, and also 3 standard splits commonly used in the literature for PDTB 2.0.

The L3 option in PDTB 3.0 preprocessing code generates the L2+L3 split used in our paper.

For PDTB 3.0, running the following will save cross-validation folds to `data/pdtb3_xval`:

      python preprocess/preprocess_pdtb3.py --data_path path/to/data --output_dir data/pdtb3_xval --split L2_xval

For PDTB 2.0, running the following will save cross-validation folds to `data/pdtb2_xval`:

      python preprocess/preprocess_pdtb2.py --data_file path/to/data/pdtb2.csv --output_dir data/pdtb2_xval --split xval

## BERT/XLNet Baselines

We used HuggingFace's pytorch-transformers for our BERT/XLNet baselines reported in the paper. This repository contains the version (plus our own modifications) that we used for the paper, but a newer version of the library is available [here](https://github.com/huggingface/transformers).

We ran our experiments under the following environment:
* Python 3.7.3
* PyTorch 1.1.0
* CUDA 9.0.176
