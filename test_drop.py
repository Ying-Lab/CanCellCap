# %%
from torch.utils.data import TensorDataset,DataLoader
from torchvision import datasets,transforms
import torch
import pandas as pd 
from models import CanCellCap as model
from utils import opt_utils,args_utils 
import importlib
import glob
import os
import numpy as np 
from data_loaders import dataloader
import matplotlib.pyplot as plt
import seaborn as sns
import json
import copy
from tqdm import tqdm
from itertools import cycle, zip_longest
from TISCH import *
import time

import torch
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
start_time = time.time()

args = args_utils.infer_args()

args.ckp = os.path.join(args.checkpoint, 'my_model.pt')
print(f"loading {args.ckp}")
args.HVG_list = torch.load(args.ckp)['HVG_list']
args.gene_num = len(args.HVG_list)
args_utils.set_random_seed(args.seed)
args.label_str = ['label', 'tissue_label', 'cancer_label']
SPL_PATH = "./data/exp1_external"
data_spl_files = []
if data_spl_files==[]:
    data_spl_files = glob.glob(os.path.join(SPL_PATH,'*.parquet'))

args.num_tissue = len(TISCH1_tissue_map.keys())
args.num_cancer_type = len(TISCH1_cancer_type_map.keys())

model_names = ['common_moe', 'common_moe_mask']

pre_result_model = []
results_path = './results/drop/'
test_drop_weight = args.test_drop_weight
test_drops = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.92,0.94,0.95,0.96,0.98]


def data_aug(data, test_drop_weight=0.05):
    mask = (torch.rand(data.size()) > test_drop_weight).float().cuda()
    data = data * mask
    return data

def plot_multiple_roc(loader, save_prefix=""):
    """
    Plot ROC curves for multiple models.

    Args:
        models (list): List of models to evaluate.
        model_names (list): List of model names corresponding to the models.
        loader: DataLoader for the dataset.
    """
    plt.figure(figsize=(10, 8))
    for test_drop_weight in test_drops:
        print(f"drop {test_drop_weight}")
        for name in  model_names:
            if name == 'common_moe_mask':
                algorithm = model.CanCellCap(args)
            elif name == 'moe_mask':
                algorithm = model.moe_mask(args)
            elif name == 'common_mask':
                algorithm = model.common_mask(args)
            elif name == 'common':
                algorithm = model.common(args)
            elif name == 'common_moe':
                algorithm = model.common_moe(args)
            elif name == 'pure_classifier':
                algorithm = model.pure_classifier(args)

            results_file = f"{results_path}{save_prefix}_{name}_drop{test_drop_weight}_result.csv"
            # if name in pre_result_model:
                # Sample data
            if os.path.exists(results_file):
                print(f'using exist {results_file}.')

                df = pd.read_csv(results_file, index_col=0)

                # Extract true labels and prediction scores
                try:
                    all_labels = df['label']
                except:
                    all_labels = df['labels']
                    df.rename(columns={'labels': 'label'}, inplace=True)
                    df.to_csv(results_file)
                y_scores = df['freq_cancer']
                all_binary_preds = df['pred'].fillna(3).astype(int)
                # Calculate accuracy and F1 score
            else:
                algorithm.load_state_dict(torch.load(f'/root/admoe_alpha/{name}/my_model.pt')['model_dict'])
                algorithm.cuda()
                algorithm.eval()

                all_labels = []
                all_preds = []
                all_binary_preds = []
                with torch.no_grad():
                    for data in loader:

                        x = data[0].cuda().float()
                        x = data_aug(x, test_drop_weight=test_drop_weight)
                        y = data[1].cuda().long()

                        logit, _, _, _ = algorithm.predict(x)
                        
                        # Collect predictions and labels
                        all_labels.extend(y.cpu().numpy())
                        all_preds.extend(logit.softmax(1).cpu().numpy())  # Use softmax probabilities
                        all_binary_preds.extend(logit.argmax(1).cpu().numpy())

                y_scores = [pred[1] for pred in all_preds]
                rows = []
                for i, (pred, label, binary_pred) in enumerate(zip(all_preds, all_labels, all_binary_preds)):
                    freq_cancer = pred[1]  # Assuming index 1 is for cancer class
                    freq_non_cancer = pred[0]  # Assuming index 0 is for non-cancer class
                    pred_label = "Cancer" if binary_pred == 1 else "Normal"  # Map binary predictions to labels
                    rows.append([i, freq_cancer, freq_non_cancer, pred_label, binary_pred, label])

                # Create DataFrame
                df = pd.DataFrame(rows, columns=['', 'freq_cancer', 'freq_non_cancer', 'pred_label', 'pred', 'label'])

                # Save to CSV
                df.to_csv(results_file, index=False)

            acc = accuracy_score(all_labels, all_binary_preds)
            f1 = f1_score(all_labels, all_binary_preds, average='weighted')
            precision = precision_score(all_labels, all_binary_preds, average='weighted')
            recall = recall_score(all_labels, all_binary_preds)

            # Calculate ROC curve and AUC for the current model
            fpr, tpr, _ = roc_curve(all_labels, y_scores, pos_label=1)
            roc_auc = auc(fpr, tpr)

            print(f"{name} - Accuracy: {acc:.4f}, recall: {recall:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, AUC: {roc_auc:.4f},")


for data_spl_file in sorted(data_spl_files):
    print(f'test {data_spl_file}')
    val_loader =  dataloader.test_dataloader(args,data_spl_file)
    for idx,loader_idx in enumerate(val_loader):
        plot_multiple_roc(val_loader[loader_idx],loader_idx)

result_map = {}

for test_drop_weight in test_drops:
    print(f"drop {test_drop_weight}")
    for model in model_names:
        dfs = []
        print(f"combining {model}")
        for filename in os.listdir(results_path):
            if 'external' in filename:
                continue
            if filename.endswith(f"{model}_drop{test_drop_weight}_result.csv"): 
                results_file = results_path+filename
                df = pd.read_csv(results_file)
                dfs.append(df)
                try:
                    all_labels = df['label'].astype(int)
                except:
                    all_labels = df['labels'].astype(int)
                    df.rename(columns={'labels': 'label'}, inplace=True)
                    df.to_csv(results_file)
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df.to_csv(f'{results_path}{model}_result_drop.csv', index=False)


    results_df = pd.DataFrame(columns=['drop_rate', 'Model', 'Accuracy', 'Recall', 'F1 Score', 'Precision', 'AUC'])

    for name in model_names:
        results_file = f'{results_path}{name}_result_drop.csv'

        # Sample data
        if not os.path.exists(results_file):
            print(f'{results_file} is not exist.')
            continue

        df = pd.read_csv(results_file, index_col=0)

        # Extract true labels and prediction scores
        try:
            all_labels = df['label'].astype(int)
        except:
            all_labels = df['labels'].astype(int)
            df.rename(columns={'labels': 'label'}, inplace=True)
            df.to_csv(results_file)

        y_scores = df['freq_cancer']
        all_binary_preds = df['pred'].fillna(2).astype(int)
        # Calculate accuracy and F1 score
        acc = accuracy_score(all_labels, all_binary_preds)
        all_binary_preds_adj = np.where(all_binary_preds == 2, 1 - all_labels, all_binary_preds)
        recall = recall_score(all_labels, all_binary_preds_adj)
        f1 = f1_score(all_labels, all_binary_preds_adj, average='weighted')
        precision = precision_score(all_labels, all_binary_preds_adj, average='weighted')


        fpr, tpr, _ = roc_curve(all_labels, y_scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        print(f"{name} - Accuracy: {acc:.4f}, recall: {recall:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, AUC: {roc_auc:.4f},")
        # Plot the ROC curve for the current model
        # Create a new DataFrame to append data
        new_row = pd.DataFrame([{
            'drop_rate': test_drop_weight,
            'Model': name,
            'Accuracy': acc,
            'Recall': recall,
            'F1 Score': f1,
            'Precision': precision,
            'AUC': roc_auc
        }])

        # Concatenate the new row with the existing DataFrame
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        
results_df.to_csv(f'{results_path}final_results_summary.csv', index=False)