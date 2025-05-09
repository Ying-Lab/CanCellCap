
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
    "mouse":["GSE155446_count","GSE136206_count","GSE119352_count"],
}
color_map = {
    "CanCell_Cap": '#4d7dff',
    "Cancer_Finder": '#b3cdff',
    "SCEVAN": '#ffd991',
    "CopyKAT": '#ffb47b',
}

models = ['CanCell_Cap','Cancer_Finder', 'SCEVAN', 'CopyKAT']
test_file_path = './mouse/'

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
        merged_df.to_csv(f'{test_file_path}{tissue_name}_{model}_result.csv', index=False)


for tissue_name in dataset_map.keys():
    results_df = pd.DataFrame(columns=['Dataset', 'Model', 'Acc', 'Recall', 'F1', 'Precision', 'AUROC'])
    plt.figure(figsize=(4.5, 3.5))

    model_name = ['CanCell_Cap', 'Cancer_Finder','SCEVAN','CopyKAT']
    non_roc_model = ['SCEVAN','CopyKAT']

    for model in model_name:
        results_file = f'{test_file_path}{tissue_name}_{model}_result.csv'

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
            print(f"{model}:{tissue_name} - Accuracy: {acc:.4f}, recall: {recall:.4f},  F1 Score: {f1:.4f}, Precision: {precision:.4f}, AUC: {roc_auc:.4f},")
            # Plot the ROC curve for the current model
            plt.plot(fpr, tpr, label=f"{model} (AUROC={roc_auc:.4f})", color=color_map[model])
        else:
            roc_auc = 0
            print(f"{model}:{tissue_name}  - Accuracy: {acc:.4f}, recall: {recall:.4f},  F1 Score: {f1:.4f}, Precision: {precision:.4f}, AUC: nan,")
    # Plot the random classifier line
    # plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")

    # Plot settings
    plt.xlabel("FPR", fontsize=16)
    plt.ylabel("TPR", fontsize=16)
    plt.title("ROC Curves", fontsize=16)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True)
    plt.show()
    plt.savefig(f'fig/roc_{tissue_name}.pdf')

