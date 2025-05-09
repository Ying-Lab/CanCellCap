import os
import sys
import time
import numpy as np
import argparse
import torch
import random

def get_time_str():
    """
    Get current time
    
    Args:
        None
    Return:
        a str, MMDD-HHMM
    """
    return time.strftime('%m%d-%H%M')

def get_args():
    """
    Get the arguments for training to run the script and store them in the args object.
    
    Args:
        None
    Return:
        args object with arguments
    """
    date_str = time.strftime('%m%d')
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_dir', type=str, default='./data/train_data/', help="path for training data")
    parser.add_argument('--val_dir', type=str, default='./data/val_data/', help="path for validating data")
    parser.add_argument('--checkpoint', type=str,default=f'test_{ date_str }', help="checkpoint dir")
    parser.add_argument('--lr', type=float, default=1.3e-3, help="learning rate") 
    parser.add_argument('--num_classes', type=int, default=2, help="number of classes")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=1, help="device id to run")
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--schusech', type=str, default='cos')
    parser.add_argument('--weight_decay', type=float, default=1.5e-4)
    parser.add_argument('--momentum', type=float,default=0.9, help='for optimizer')
    parser.add_argument('--max_epoch', type=int,default=50, help="max iterations")
    parser.add_argument('--genelist_outname', type=str,default='gene_list.txt', help="genelist_outname")
    parser.add_argument('--logs_name', type=str,default='train_log', help="train_log")
    parser.add_argument('--batch_size', type=int,default=128, help="batch_size")
    parser.add_argument('--weight_tissue', type=float,default=0.3, help="weight of moe routing loss")
    parser.add_argument('--weight_ad_train', type=float,default=0.2, help="weight of common ad loss")
    parser.add_argument('--gene_drop_weight', type=float,default=0.3, help="weight of drop ratio")
    parser.add_argument('--weight_reconstruct', type=float,default=0.1, help="weight of reconstruct loss")
    parser.add_argument('--weight_type', type=float,default=0, help="for cancer origin loss")
    parser.add_argument('--model_name', type=str,default='common_moe_mask', help="model_name:common_mask\common_moe\moe_mask\common_moe_mask")
    parser.add_argument('--species', type=str,default='human', help="human or mouse")
    args = parser.parse_args()

    return args

def infer_args():
    """
    Get the arguments for inference to run the script and store them in the args object.
    
    Args:
        None
    Return:
        args object with arguments
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_classes', type=int, default=2, help="number of classes")
    parser.add_argument('--test_dir', type=str, default='./data/exp1_external/', help="path for testing data")
    parser.add_argument('--test_file', type=str, default='/root/CanCellCap/data/exp1_external/Brain_gene_data.parquet', help="path for testing data")
    parser.add_argument('--checkpoint', type=str,default='./ckpt/', help="output dir")
    parser.add_argument('--genelist_outname', type=str,default='gene_list.txt', help="genelist_outname")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--label_str', type=str,default='label', help="the row label for the label in the training data")
    parser.add_argument('--batch_size', type=int,default=512, help="batch_size")
    parser.add_argument('--gpu_id', type=int, default=0, help="device id to run")
    parser.add_argument('--model_name', type=str,default='common_moe_mask', help="model_name:common_mask\common_moe\moe_mask\common_moe_mask")
    parser.add_argument('--test_drop_weight', type=float,default=0, help="weight of test drop ratio")
    parser.add_argument('--species', type=str,default='human', help="human or mouse")
    parser.add_argument('--time_mode', type=str,default='read_infer', help="read_infer or infer")
    parser.add_argument('--output', type=str,default='./results/', help="Result output location")
    parser.add_argument('--modin', type=bool,default=False, help="Read the file using modin.")
    parser.add_argument('--explainer', type=str,default='shap',choices=['ig', 'deeplift', 'gradientshap', 'shap'],help="Select the interpreter method: 'ig' (Integrated Gradients), 'deeplift', 'gradientshap', 'shap'")
    args = parser.parse_args()
    return args


def set_random_seed(seed=0):
    """
    set the random seeds
    
    Args:
        seed (int): random seed
    Return:
        None
    """
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_environ():
    """
    Print the current environment and package version

    Returns:
        None
    """
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))



