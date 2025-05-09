
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
    "unseen cancer":["GSE150766","GSE168652_RAW","GSE180665_gene_data","SCP1415_count"],
    "unseen tissue":["GSE137829_RAW",'GSE176031_RAW','GSE141445_RAW',"GSE163678_RAW"],
    # "rare_cancer":["acp_data",'GSE131309_count','GSE150766',"GSE159115_KICH",'GSE163678_RAW','SCP2393'],
    # "under_studied tissue": ["GSE138433_RAW","GSE131309_count","GSE163678_RAW","GSE159115_KICH"],
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
    "CTC":["GSE109761_gene_data"],
    "UVM":["GSE138433_RAW"],
    "PAAD":["GSE141017_RAW","GSE165399_RAW"],
    "Glioma" :["GSE141383_RAW","GSE141460_count"],
    "NSCLC" : ["GSE148071_RAW","GSE150660_RAW"],
    "LIHC":["GSE146115_RAW"],
    "OV" : ["GSE147082_count","Pelvic_cavity_gene_data","GSE130000_RAW"],
    "ALL":["GSE154109_RAW"],
    "MB":["GSE155446_gene_data"],
    "CC":["GSE168652_RAW"],
    "MM":["GSE175385_RAW"],
    "PRAD" : ["GSE176031_RAW","GSE137829_RAW","GSE141445_RAW"],
    "HNSC":["GSE180268_gene_data"],
    "LUAD":["GSM3618014_gene_data"],
    "BRCA":["Breast_gene_data"],
    "CRC":["Colorectal_gene_data"],
    "LUNG":["Lung_gene_data"],
    "CMM" : ["SCP1415_count"],
    "ACP":["acp_data"],
    "SS":["GSE131309_count"],
    "HB":["GSE180665_gene_data"],
    "PPB":["GSE163678_RAW"],
    "KICH":["GSE159115_KICH"],
    "SCLC":["GSE150766"],
    "GEP-NETs":["SCP2393"],
}


all_values = []
for key in dataset_map:
    all_values.extend(dataset_map[key])

all_values = sorted(set(all_values))

dataset_map['all'] =  all_values

color_map = {
    "CanCellCap": '#4d7dff',
    "Cancer_Finder": '#b3cdff',
    "PreCanCell": '#8aedff',
    "SCEVAN": '#ffd991',
    "CopyKAT": '#ffb47b',
    "ikarus": '#ff852e'
}

models = ['CanCellCap','Cancer_Finder','PreCanCell', 'SCEVAN', 'CopyKAT', 'ikarus']
test_file_path = './all_results/'

for tissue_name in dataset_map.keys():
    for model in models:
        dfs = []
        # print(f"combining {model}")
        for filename in dataset_map[tissue_name]:
            result_file = f"{test_file_path}{filename}_{model}_result.csv"
            if os.path.exists(result_file):
                dfs.append(pd.read_csv(result_file))
            else:
                print(f"miss file:{result_file}")
        if dfs == []:
            continue
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df.to_csv(f'./combine/{tissue_name}_{model}_result.csv', index=False)

results_df = pd.DataFrame()
for tissue_name in dataset_map.keys():
    plt.figure(figsize=(4.5, 3.5))

    model_name = ['CanCellCap', 'Cancer_Finder','PreCanCell','SCEVAN','CopyKAT','ikarus']
    non_roc_model = ['SCEVAN','CopyKAT']

    for model in model_name:
        results_file = f'./combine/{tissue_name}_{model}_result.csv'

        # Sample data
        if not os.path.exists(results_file):
            print(f'{results_file} is not exist.')
            continue

        df = pd.read_csv(results_file, index_col=0)

        all_labels = df['label'].astype(int)

        y_scores = df['freq_cancer']
        all_binary_preds = df['pred'].fillna(2).astype(int)
        # Calculate accuracy and F1 score
        acc = accuracy_score(all_labels, all_binary_preds)
        all_binary_preds_adj = np.where(all_binary_preds == 2, 1 - all_labels, all_binary_preds)
        recall = recall_score(all_labels, all_binary_preds_adj)
        f1 = f1_score(all_labels, all_binary_preds_adj, average='weighted')
        precision = precision_score(all_labels, all_binary_preds_adj, average='weighted')
        if all_labels.sum() == 0:
            f1, reacall, precision, roc_auc = 0,0,0,0
        # Calculate ROC curve
        if model not in non_roc_model:
            fpr, tpr, _ = roc_curve(all_labels, y_scores, pos_label=1)
            roc_auc = auc(fpr, tpr)
            print(f"{model}:{tissue_name} - Accuracy: {acc:.4f}, recall: {recall:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, AUC: {roc_auc:.4f},")
            # Plot the ROC curve for the current model
            plt.plot(fpr, tpr, label=f"{model} (AUROC={roc_auc:.4f})", color=color_map[model])
        else:
            roc_auc = 0
            print(f"{model}:{tissue_name}  - Accuracy: {acc:.4f}, recall: {recall:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, AUC: nan,")
    
        new_row = pd.DataFrame([{
            'Dataset': tissue_name,
            'Model': model,
            'Accuracy': acc,
            'Recall': recall,
            'F1 Score': f1,
            'Precision': precision,
            'AUC': roc_auc
        }])

        results_df = pd.concat([results_df, new_row])

    # Plot settings
    plt.xlabel("FPR", fontsize=16)
    plt.ylabel("TPR", fontsize=16)
    plt.title("ROC Curves", fontsize=16)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True)
    plt.show()
    plt.savefig(f'fig/roc_{tissue_name}.pdf')


results_df.to_csv(f'./average.csv', index=False)
