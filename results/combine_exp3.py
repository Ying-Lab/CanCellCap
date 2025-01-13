
import pandas as pd
import os
import scanpy as sc
from collections import defaultdict

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score, precision_score, recall_score
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'

dataset_map = {
    "novel_cancer":["GSE168652_RAW","GSE180665_RAW"],
    "novel_tissue":["GSE163678_RAW","GSE176031_RAW"],
}

models = ['CanCell_Cap','Cancer_Finder','pure_classifier','PreCanCell', 'ikarus', 'SCEVAN', 'CopyKAT']
test_file_path = './exp2_unseen/'

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
        merged_df.to_csv(f'{test_file_path}{tissue_name}_{model}_result_exp3.csv', index=False)


for tissue_name in dataset_map.keys():
    results_df = pd.DataFrame(columns=['Dataset', 'Model', 'Accuracy', 'Recall', 'F1 Score', 'Precision', 'AUC'])
    plt.figure(figsize=(10, 8))

    model_name = ['CanCell_Cap', 'Cancer_Finder','PreCanCell','SCEVAN','CopyKAT','ikarus']
    non_roc_model = ['SCEVAN','CopyKAT']
    for model in model_name:
        results_file = f'{test_file_path}{tissue_name}_{model}_result_exp3.csv'

        # Sample data
        if not os.path.exists(results_file):
            print(f'{results_file} is not exist.')
            continue

        df = pd.read_csv(results_file, index_col=0)

        all_labels = df['label'].astype(int)

        y_scores = df['freq_cancer']
        all_binary_preds = df['pred'].fillna(3).astype(int)
        # Calculate accuracy and F1 score
        acc = accuracy_score(all_labels, all_binary_preds)
        f1 = f1_score(all_labels, all_binary_preds, average='weighted')
        precision = precision_score(all_labels, all_binary_preds, average='weighted')
        # breakpoint()
        pre_index = all_binary_preds != 2
        recall = recall_score(all_labels[pre_index], all_binary_preds[pre_index])#, average='weighted')
        specificity = recall_score(all_labels[pre_index], all_binary_preds[pre_index], pos_label=0)

        # Calculate ROC curve
        if model not in non_roc_model:
            fpr, tpr, _ = roc_curve(all_labels, y_scores, pos_label=1)
            roc_auc = auc(fpr, tpr)
            print(f"{model}:{tissue_name} - Accuracy: {acc:.4f}, recall: {recall:.4f}, specificity: {specificity:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, AUC: {roc_auc:.4f},")
            # Plot the ROC curve for the current model
            plt.plot(fpr, tpr, label=f"{model} (AUC = {roc_auc:.4f})")
        else:
            print(f"{model}:{tissue_name}  - Accuracy: {acc:.4f}, recall: {recall:.4f}, specificity: {specificity:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, AUC: nan,")
    # Plot the random classifier line
    # plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")

    # Plot settings
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curves", fontsize=14)
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    plt.savefig(f'fig/exp2_roc_{tissue_name}.svg')

