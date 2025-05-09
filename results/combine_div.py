
import pandas as pd
import os
import scanpy as sc
from collections import defaultdict
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score, precision_score, recall_score
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'
dataset_map = {
    "novel_cancer":["GSE150766","GSE168652_RAW","GSE180665_gene_data","SCP1415_count"],
    "novel_tissue":["GSE137829_RAW",'GSE176031_RAW','GSE141445_RAW',"GSE163678_RAW"],
    "rare_cancer":["acp_data",'GSE131309_count','GSE150766',"GSE159115_KICH",'GSE163678_RAW','SCP2393'],
    "under_studied tissue": ["GSE138433_RAW","GSE131309_count","GSE163678_RAW","GSE159115_KICH"],
    "Smart-seq2":["GSE109761_gene_data","GSE141460_count"],
    "10X Genomics":["GSE180665_gene_data","GSE175385_RAW"],
    "Microwell-seq":["GSE141383_RAW"],
    "C1 Fluidigm":["GSE146115_RAW"],
    "Drop-seq":["GSE130000_RAW","GSE147082_count"],
    "GEXSCOPETM":["GSE148071_RAW","GSE165399_RAW"],
    'Blood': ["GSE154109_RAW","PBMCs_gene_data"],
    'Bone': ["GSE175385_RAW"],
    'Brain': ["acp_data","GSE141383_RAW","GSE141460_count","GSE155446_gene_data"],
    'Breast': ["Breast_gene_data","GSE109761_gene_data"],
    'Colorectal': ["Colorectal_gene_data"],
    'Head and neck': ["GSE180268_gene_data"],
    'Liver': ["GSE146115_RAW","GSE180665_gene_data"],
    'Lung': ["Lung_gene_data"],
    'Pancreas': ["GSE141017_RAW","GSE165399_RAW",'SCP2393'],
    'Pelvic cavity': ["GSE130000_RAW","GSE147082_count","GSE168652_RAW","Pelvic_cavity_gene_data"],
    'Prostate': ["GSE137829_RAW",'GSE141445_RAW','GSE176031_RAW'],
    'Skin': ["SCP1415_count"],
    'Eye': ["GSE138433_RAW"],
    'Kidney': ["GSE159115_KICH"],
    'Soft tissue': ["GSE131309_count","GSE163678_RAW"],
}


color_map = {
    "CanCellCap": '#4d7dff',
    "Cancer_Finder": '#b3cdff',
    "PreCanCell": '#8aedff',
    "SCEVAN": '#ffd991',
    "CopyKAT": '#ffb47b',
    "ikarus": '#ff852e'
}

models = ['CanCellCap','Cancer_Finder','PreCanCell', 'SCEVAN', 'CopyKAT', 'ikarus']
non_roc_model = ['SCEVAN','CopyKAT']
test_file_path = './all_results/'

results_df = pd.DataFrame()
for filename in dataset_map['all_results']:
    new_row = pd.DataFrame([{'Dataset': filename}])
    print(f'{filename}')
    for model in models:
        result_file = f"{test_file_path}{filename}_{model}_result.csv"
        
        if not os.path.exists(result_file):
            print(f'{result_file} is not exist.')
            breakpoint()
            continue
        df = pd.read_csv(result_file)
        
        all_labels = df['label']

        y_scores = df['freq_cancer']
        all_binary_preds = df['pred'].fillna(2).astype(int)
        # Calculate accuracy and F1 score
        acc = accuracy_score(all_labels, all_binary_preds)
        reject_num = (all_binary_preds == 2).sum()
        all_binary_preds_adj = np.where(all_binary_preds == 2, 1 - all_labels, all_binary_preds)

        recall = recall_score(all_labels, all_binary_preds_adj)
        f1 = f1_score(all_labels, all_binary_preds_adj, average='weighted')
        precision = precision_score(all_labels, all_binary_preds_adj, average='weighted')
        if all_labels.sum() == 0:
            f1, reacall, precision, roc_auc = 0,0,0,0

        if model not in non_roc_model:
            fpr, tpr, _ = roc_curve(all_labels, y_scores, pos_label=1)
            roc_auc = auc(fpr, tpr)
            print(f"{model} - Accuracy: {acc:.4f}, recall: {recall:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, AUC: {roc_auc:.4f},")
            # Plot the ROC curve for the current model
            plt.plot(fpr, tpr, label=f"{model} (AUC = {roc_auc:.4f})")
        else:
            roc_auc = 0
            print(f"{model} - Accuracy: {acc:.4f}, recall: {recall:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, AUC: NA,")
        
        # print(f"reject number = {reject_num}, ratios = {reject_num/len(all_labels)}")

        result_row = pd.DataFrame([{
            f'{model} Accuracy': acc,
            f'{model} Recall': recall,
            f'{model} F1 Score': f1,
            f'{model} Precision': precision,
            f'{model} AUC': roc_auc
        }])
        new_row = pd.concat([new_row, result_row], axis=1)

    results_df = pd.concat([results_df, new_row])

metrics = ['Accuracy', 'Recall', 'F1 Score', 'Precision', 'AUC']
index = ['Dataset']
for metric in metrics:
    for model in models:
        index.append(f'{model} {metric}')

results_df = results_df[index]
results_df.to_csv(f'./all_results.csv', index=False)

