import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score, precision_score, recall_score

models = ['CanCell_Cap','Cancer_Finder','PreCanCell', 'ikarus', 'SCEVAN', 'CopyKAT']
test_file_path = './exp1_external/'

for model in models:
    dfs = []
    print(f"combining {model}")
    for filename in os.listdir(test_file_path):
        if 'external' in filename:
            continue
        if filename.endswith(f"{model}_result.csv"): 
            results_file = test_file_path+filename
            df = pd.read_csv(results_file)
            dfs.append(df)
            try:
                all_labels = df['label'].astype(int)
            except:
                all_labels = df['labels'].astype(int)
                df.rename(columns={'labels': 'label'}, inplace=True)
                df.to_csv(results_file)
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(f'{test_file_path}{model}_result_exp1.csv', index=False)


results_df = pd.DataFrame(columns=['Dataset', 'Model', 'Accuracy', 'Recall', 'F1 Score', 'Precision', 'AUC'])
plt.figure(figsize=(10, 8))

model_name = ['CanCell_Cap', 'Cancer_Finder','PreCanCell','SCEVAN','CopyKAT','ikarus']
non_roc_model = ['SCEVAN','CopyKAT']
for name in model_name:
    results_file = f'{test_file_path}{name}_result_exp1.csv'

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
    f1 = f1_score(all_labels, all_binary_preds, average='weighted')
    precision = precision_score(all_labels, all_binary_preds, average='weighted')
    try:
        pre_index = all_binary_preds != 2
        recall = recall_score(all_labels[pre_index], all_binary_preds[pre_index])#, average='weighted')
        specificity = recall_score(all_labels[pre_index], all_binary_preds[pre_index], pos_label=0)
    except:
        breakpoint()

    # Calculate ROC curve
    if name not in non_roc_model:
        fpr, tpr, _ = roc_curve(all_labels, y_scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        print(f"{name} - Accuracy: {acc:.4f}, recall: {recall:.4f}, specificity: {specificity:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, AUC: {roc_auc:.4f},")
        # Plot the ROC curve for the current model
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")
    else:
        print(f"{name} - Accuracy: {acc:.4f}, recall: {recall:.4f}, specificity: {specificity:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, AUC: nan,")
# Plot the random classifier line
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")

# Plot settings
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Multiple Models")
plt.legend(loc="best")
plt.grid(True)
plt.show()
plt.savefig(f'fig/exp1_roc_total.pdf')


