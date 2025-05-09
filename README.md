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
The training, validation, and testing datasets processed in this study are available for download from [Google Drive](https://drive.google.com/file/d/1phVCEXbi0Azl3NGj273aemEXp2E0tZWH/view?usp=drive_link)

To test CanCellCap, place parquet files in the *data* directory.
### Ckpt download
A pretrained CanCellCap model is provided in this repository for immediate use. It allows users to quickly test the framework on their own scRNA-seq data without the need for extensive training.

Detailed instructions for downloading and using the pretrained model can be found in [Google Drive](https://drive.google.com/file/d/1Vgh_hOYGqwEu1jRhdIdvT6gAzb7gy28f/view?usp=drive_link)

After downloading, extract the compressed file and place *my_model.pt* into the *ckpt* directory.
### Test example
This script is used to perform inference with the trained CanCellCap model on scRNA-seq data.
```
python infer.py --test_file ./data/Pelvic_cavity_100.csv --checkpoint ./ckpt/ --output ./
```

Parameter Description
- --test_file: Path to the test data file (e.g., .csv or .parquet).

- --checkpoint: Directory where the pre-trained model weights are saved.
- --batch_size: Batch size for inference (default is 512).
- --gpu_id: GPU device ID to run the program (default is 0).
- --species: Species of the data, either 'human' or 'mouse' (default is 'human').
- --output: Directory where the results will be saved.
- --modin: Whether to use Modin for file reading (default is False); recommended for large CSV files.


### Train example
This script is used to train the CanCellCap model with configurable parameters for optimization, regularization, and loss balancing.
```
python train.py --train_dir ./data/train_data/ --val_dir ./data/val_data/ --checkpoint ckpt_1 --max_epoch 50
```
Parameter Description
- --train_dir: Path to the training data directory (default is './data/train_data/').

- --val_dir: Path to the validation data directory (default is './data/test_data/').

- --checkpoint: Directory where the pre-trained model weights are saved (default is 'test_{date_str}').

- --lr: Learning rate for the optimizer (default is 1.3e-3).

- --seed: Random seed value for reproducibility (default is 0).

- --gpu_id: GPU device ID to run the program (default is 1).

- --weight_decay: Weight decay for regularization (default is 1.5e-4).

- --momentum: Momentum for the optimizer (default is 0.9).

- --max_epoch: Maximum number of training epochs (default is 50).

- --batch_size: Batch size for training (default is 128).

- --weight_tissue: Weight of the MOE routing loss (default is 0.3).

- --weight_ad_train: Weight of the common adversarial loss during training (default is 0.2).

- --gene_drop_weight: Weight of the gene drop ratio (default is 0.3).

- --weight_reconstruct: Weight of the reconstruction loss (default is 0.1).

- --weight_type: Weight for cancer origin loss (default is 0).



## Reproducing Experiments from the Paper

To reproduce the experiments corresponding to each section of the paper, use the following commands:


### Experiments for performance
```
# identify cancer cell on 33 testing datasets
python test_testing.py
# Analyze results by tissue types, cancer types, sequencing platforms, as well as unseen cancer and tissue types
python results/combine.py
```

### Experiments for performance of mouse dataset
```
python test_mouse.py
```

### Spot-level cancer identification in spatial transcriptomics
```
python test_st.py
```

