import torch
import os
import pandas as pd 
import scanpy
import anndata
import glob
from torch.utils.data import TensorDataset,DataLoader
import gc
import csv

import torch
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt


def accuracy(network, loader):
    """
    Calculate the accuracy of the network in the dataset

    Args:
        network : the networks that need to be evaluated
        loader: data_loader for the dataset
    Returns:
        accuracy 
    """
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            # if self.args.gpu_id:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            y_tissue = data[2].cuda().long()
            y_type = data[3].cuda().long()
            logit, logit_tissue, logit_common_domain, _ = network.predict(x)
            p = logit.argmax(1)
            correct += (p.eq(y).float()).sum().item()
            total += len(y)

    network.train()
    return correct / total

def accuracy_class(network, loader, class_value):
    """
    Calculate the accuracy of the network in the dataset

    Args:
        network : the networks that need to be evaluated
        loader: data_loader for the dataset
    Returns:
        accuracy 
    """
    correct = 0
    tissue_correct = 0
    type_correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            # if self.args.gpu_id:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            y_tissue = data[2].cuda().long()
            y_type = data[3].cuda().long()

            logit, logit_tissue, logit_type = network.predict(x)
            p = logit.argmax(1)
            p_type = logit_type.argmax(1) * p
            p_tissue = logit_tissue.argmax(1)

            mask = y==class_value
            p = p[mask]
            p_tissue = p_tissue[mask]
            p_type = p_type[mask]
            y = y[mask]
            y_tissue = y_tissue[mask]
            y_type = y_type[mask]

            correct += (p.eq(y).float()).sum().item()
            tissue_correct += (p_tissue.eq(y_tissue).float()).sum().item()
            type_correct += (p_type.eq(y_type).float()).sum().item()
            total += len(y)
    # print(f"infer cell number: {total}")
    network.train()
    if total == 0:
        return 0, 0, 0
    return correct / total, tissue_correct / total, type_correct/total

def accuracy_mlp(network, loader):
    """
    Calculate the accuracy of the network in the dataset

    Args:
        network : the networks that need to be evaluated
        loader: data_loader for the dataset
    Returns:
        accuracy 
    """
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            # if self.args.gpu_id:
            x = data[0].cuda().float()
            y = data[1].cuda().long()

            logit, logit_tissue, logit_type = network.predict(x)
            p = logit.argmax(1)

            correct += (p.eq(y).float()).sum().item()
            total += len(y)

    network.train()
    return correct / total


def accuracy_threshold(network, loader, threshold=0.5):
    """
    Calculate the accuracy of the network in the dataset

    Args:
        network : the networks that need to be evaluated
        loader: data_loader for the dataset
    Returns:
        accuracy 
    """
    correct = 0
    tissue_correct = 0
    type_correct = 0
    total = 0
    if threshold>1 or threshold < 0:
        print(f"error threshold:{threshold}")
        return 0,0,0

    network.eval()
    with torch.no_grad():
        for data in loader:
            # if self.args.gpu_id:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            y_tissue = data[2].cuda().long()
            y_type = data[3].cuda().long()

            logit, logit_tissue, logit_type = network.predict(x)

            p = logit.argmax(1)
            p_type = logit_type.argmax(1) * p
            p_tissue = logit_tissue.argmax(1)
            # 拒绝预测置信度为[0.5-threshold, 0.5+threshold]中的值

            logit_max = logit.max(dim=1).values
            mask = (logit_max >= threshold)

            p = p[mask]
            p_tissue = p_tissue[mask]
            p_type = p_type[mask]
            y = y[mask]
            y_tissue = y_tissue[mask]
            y_type = y_type[mask]

            correct += (p.eq(y).float()).sum().item()
            tissue_correct += (p_tissue.eq(y_tissue).float()).sum().item()
            type_correct += (p_type.eq(y_type).float()).sum().item()
            total += len(y)
    print(f"infer cell number: {total}. logit's threshold: {threshold}")
    network.train()
    if total == 0:
        return 0, 0, 0
    return correct / total, tissue_correct / total, type_correct/total

def save_checkpoint(filename, alg, args):
    """
    Save the checkpoint of the pre-trained model.

    Args:
        filename: Name of the target file to be saved
        alg: neural network models that need to be saved
        args: training parameters

    Returns:
        None 
    """
    save_dict = {
        "model_dict": alg.cpu().state_dict(),
        "HVG_list": args.HVG_list
    }
    torch.save(save_dict, os.path.join(args.checkpoint, filename))

def normalize_matrix_counts(raw_df,HVG_list,target_sum=10000):
    """
    Normalize raw matrix counts and select feature genes

    Args:
        raw_df: The matrix that need to be normalized, pandas.Dataframe format 
        HVG_list: A list of feature genes

    Returns:
        A matrix after normalization and selection of feature genes, pandas.Dataframe format 
    """
    gene_list = pd.DataFrame(HVG_list).set_index(0)
    raw_df['sum'] = raw_df.sum(axis=1)
    raw_df = raw_df.sort_values(by='sum',ascending=False)
    raw_df = raw_df.drop(columns='sum')
    raw_df = raw_df[~raw_df.index.duplicated()]
    raw_df = raw_df.T
    
    
    adata = anndata.AnnData(raw_df,raw_df.index.to_frame(), raw_df.columns.to_frame())
    scanpy.pp.normalize_total(adata,target_sum=target_sum)
    scanpy.pp.log1p(adata)
    raw_df = pd.DataFrame(adata.X,index=adata.obs.index,columns=adata.var.index)
    select_df = pd.merge(gene_list,raw_df.T,how='left',left_index=True,right_index=True)
    select_df = select_df.fillna(0.0).T[gene_list.index]
    return select_df


def generate_genelist(args):
    """
    Generate a list of genes to be used as features

    Args:
        args: training parameters

    Returns:
        A list of feature genes
    """

    fst_flag = True
    merged_df = pd.DataFrame()
    for file_name in glob.glob(os.path.join(args.train_dir,'*')):
        if file_name.endswith('.tsv'):
            raw_df = pd.read_csv(file_name,index_col=0,sep='\t')
        elif file_name.endswith('.csv'):
            raw_df = pd.read_csv(file_name,index_col=0,sep=',')
        elif file_name.endswith('.h5ad'):
            raw_df = scanpy.read_h5ad(file_name).to_df()
        
        raw_df['sum'] = raw_df.sum(axis=1)
        raw_df = raw_df.sort_values(by='sum',ascending=False)
        raw_df = raw_df.drop(columns='sum')
        raw_df = raw_df[~raw_df.index.duplicated()]
        
        if fst_flag:
            merged_df = raw_df.copy()
        else :
            merged_df = pd.merge(merged_df,raw_df,left_index=True,right_index=True,how='inner')
    merged_df = merged_df[merged_df.index!=args.label_str]
    merged_df['var'] = merged_df.var(axis=1)
    merged_df = merged_df.sort_values(by='var',ascending=False)
    genelist = list(merged_df.index[:args.gene_num])
    pure_genelist = genelist.copy()
    genelist.append(args.label_str)
    pd.DataFrame(index=genelist).to_csv(os.path.join(args.output,args.genelist_outname),header=None)
    return pure_genelist

class InferLoaders():
    """
    Iterators for the data_loader for inference

    Args:
        args: inference parameters
        step_num : int, the sample number of one s

    Returns:
        input_data : a dataframe with barcodes for inference
        input_loader : a data loader for inference
    """
    def __init__(self,args,step_num = 1e4):
        self.obj_filename = args.matrix
        self.idx = 1
        self.step_num = step_num
        self.args = args
    def __iter__(self):
    
        if self.obj_filename.endswith('.h5ad'):
            self.full_raw_df = scanpy.read_h5ad(self.obj_filename).to_df()
            self.row = self.full_raw_df.columns
            self.row_len = len(self.row)

        elif self.obj_filename.endswith('.csv'):
            with open(self.obj_filename,'r') as csv_file:
                reader = csv.reader(csv_file,delimiter=',')
                self.row = reader.__iter__().__next__()
            self.row_len = len(self.row)
            

        elif self.obj_filename.endswith('.tsv'):
            with open(self.obj_filename,'r') as csv_file:
                reader = csv.reader(csv_file,delimiter='\t')
                self.row = reader.__iter__().__next__()
            self.row_len = len(self.row)
        else :
            raise ValueError("Unrecognized Formats")
        self.idx = 1
        return self
    def __next__(self):
        if self.idx < self.row_len:
            step_end = int(min(self.idx + self.step_num,self.row_len))
            if self.obj_filename.endswith('.h5ad'):
                raw_df = self.full_raw_df.iloc[:,self.idx:step_end]
            elif self.obj_filename.endswith('.csv'):
                print(self.idx,step_end)
                col_index = [self.row[0]] + self.row[self.idx:step_end]
                raw_df = pd.read_csv(self.obj_filename,index_col=0,usecols=col_index,sep=',')

            elif self.obj_filename.endswith('.tsv'):
                col_index = [self.row[0]] + self.row[self.idx:step_end]
                raw_df = pd.read_csv(self.obj_filename,index_col=0,usecols=col_index,sep='\t')
            else:
                raise ValueError("Unrecognized Formats")
            input_data = normalize_matrix_counts(raw_df,self.args.HVG_list)
            input_set = TensorDataset(torch.from_numpy(input_data.values).float())
            input_loader = DataLoader(dataset=input_set,batch_size = len(input_set)) 
            self.idx = int(self.idx + self.step_num)
            del raw_df
            gc.collect()
            return input_data,input_loader

        else :
            raise StopIteration
        


def get_optimizer(alg, args):
    """
    Get the optimizer for training

    Args:
        args: training parameters
        alg : networks model

    Returns:
        optimizer for training
        
    """
    params = alg.parameters()
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    return optimizer

def get_scheduler(optimizer, args):
    """
    Get the scheduler for training

    Args:
        args: training parameters
        alg : networks model

    Returns:
        scheduler for training
        
    """
    if not args.schuse:
        return None
    if args.schusech == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.max_epoch * args.steps_per_epoch)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler