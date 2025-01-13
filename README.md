# CanCellCap

## Overview
CanCellCap is a multi-domain learning framework designed to identify cancer cells from single-cell RNA sequencing (scRNA-seq) data, suitable for all tissues, cancers, and sequencing platforms. By integrating domain adversarial learning and Mixture of Experts (MoE), CanCellCap extracts both tissue-common and tissue-specific patterns from gene expression profiles. Its masking-reconstruction strategy addresses sequencing platform variability, ensuring robust performance across diverse datasets.

## Installation
CanCellCap is a Python package. To install it, run the following command in your terminal.
1. Clone the repository and navigate to the project directory:
```
git clone https://github.com/Ying-Lab/CanCellCap.git && cd CanCellCap
```

2. Create a conda environment with Python 3.9+ and configure it using requirements.txt after activating the environment.
```
conda create -n CanCellCap python=3.9.19
conda activate CanCellCap
pip install -r requirements.txt
```

## Run 
### Data preparation
The training and validation datasets can be download from:[]

All testing datasets can be downloaded from:[]

### Ckpt download
A pretrained CanCellCap model is provided in this repository for immediate use. It allows users to quickly test the framework on their own scRNA-seq data without the need for extensive training.

Detailed instructions for downloading and using the pretrained model can be found in the xxx[lianjie]

### Test example
```
python test.py --val_dir ./data --checkpoint ./ckpt/
```

### Train example

```
python train.py --train_dir ./data/train_data/ --val_dir ./data/test_data/ --checkpoint ckpt_1 --max_epoch 50
```
