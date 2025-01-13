
from models import CanCellCap
import torch
import pandas as pd 
from utils import opt_utils,args_utils 
import glob
import os
from data_loaders import dataloader
import matplotlib.pyplot as plt
from TISCH import *
import torch
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'

args = args_utils.infer_args()

args.ckp = os.path.join(args.output, 'my_model.pt')

args.HVG_list = torch.load(args.ckp)['HVG_list']
args.gene_num = len(args.HVG_list)
args_utils.set_random_seed(args.seed)
args.label_str = ['label', 'tissue_label', 'cancer_label']
args.val_dir = './data/exp3_platforms'
SPL_PATH = args.val_dir
# data_spl_files = [os.path.join(SPL_PATH,"Bond.csv")]
data_spl_files = []
if data_spl_files==[]:
    data_spl_files = glob.glob(os.path.join(SPL_PATH,'*.parquet'))

args.num_tissue = len(TISCH1_tissue_map.keys())
args.num_cancer_type = len(TISCH1_cancer_type_map.keys())

model_names = ['common', 'common_mask', 'common_moe', 'moe_mask', 'common_moe_mask']

results_path = './results/exp3_platforms/'


def CanCell_Cap():
    CanCell_Cap = CanCellCap.CanCellCap(args)
    CanCell_Cap.load_state_dict(torch.load(args.ckp)['model_dict'])
    print(f"loading {args.ckp}")
    CanCell_Cap.cuda()
    CanCell_Cap.eval()
    return CanCell_Cap


def plot_multiple_roc(loader, dataset_name=""):
    """
    Plot ROC curves for multiple models.

    Args:
        models (list): List of models to evaluate.
        loader: DataLoader for the dataset.
    """
    plt.figure(figsize=(10, 8))
    cancell_cap_model = CanCell_Cap()
    
    models = {'CanCell_Cap':cancell_cap_model, 'Cancer_Finder':'Cancer_Finder', 'PreCanCell':'precancell','SCEVAN':'SCEVAN','CopyKAT':'CopyKAT', 'ikarus':'ikarus'}

    pre_result_model = ['PreCanCell','SCEVAN','CopyKAT','ikarus','Cancer_Finder']
    non_roc_model = ['SCEVAN','CopyKAT']

    results_df = pd.DataFrame(columns=['Dataset', 'Model', 'Accuracy', 'Recall', 'F1 Score', 'Precision', 'AUC'])

    for name in models.keys():
        results_file = f'{results_path}{dataset_name}_{name}_result.csv'
        if name in pre_result_model:
            # Sample data
            if not os.path.exists(results_file):                                                                                                                                                                                                                                                      
                print(f'{results_file} is not exist.')
                continue

            df = pd.read_csv(results_file)

            # Extract true labels and prediction scores
            try:
                all_labels = df['label']
            except:
                all_labels = df['labels']
                df.rename(columns={'labels': 'label'}, inplace=True)
                df.to_csv(results_file)
            y_scores = df['freq_cancer']
            all_binary_preds = df['pred']
            # Calculate accuracy and F1 score
            acc = accuracy_score(all_labels, all_binary_preds)
            f1 = f1_score(all_labels, all_binary_preds, average='weighted')
            precision = precision_score(all_labels, all_binary_preds, average='weighted')
            pre_index = all_binary_preds != 2
            recall = recall_score(all_labels[pre_index], all_binary_preds[pre_index])#, average='weighted')
            # Calculate ROC curve
            if name not in non_roc_model:
                fpr, tpr, _ = roc_curve(all_labels, y_scores, pos_label=1)
                roc_auc = auc(fpr, tpr)
                print(f"{name} - Accuracy: {acc:.4f}, recall: {recall:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, AUC: {roc_auc:.4f},")
                # Plot the ROC curve for the current model
                plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")
            else:
                print(f"{name} - Accuracy: {acc:.4f}, recall: {recall:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, AUC: nan,")

        else:
            all_labels = []
            all_preds = []
            all_binary_preds = []
            with torch.no_grad():
                for data in loader:

                    x = data[0].cuda().float()
                    y = data[1].cuda().long()

                    logit = models[name].infer(x)
                    # Collect predictions and labels
                    all_labels.extend(y.cpu().numpy())
                    all_preds.extend(logit.softmax(1).cpu().numpy())  # Use softmax probabilities
                    all_binary_preds.extend(logit.argmax(1).cpu().numpy())  # Predicted labels for accuracy, F1, precision

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

            # Calculate evaluation metrics
            acc = accuracy_score(all_labels, all_binary_preds)
            f1 = f1_score(all_labels, all_binary_preds, average='weighted')
            precision = precision_score(all_labels, all_binary_preds, average='weighted')
            recall = recall_score(all_labels, all_binary_preds)#, average='weighted')

            # Calculate ROC curve and AUC for the current model
            fpr, tpr, _ = roc_curve(all_labels, [pred[1] for pred in all_preds], pos_label=1)
            roc_auc = auc(fpr, tpr)

            print(f"{name} - Accuracy: {acc:.4f}, recall: {recall:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, AUC: {roc_auc:.4f},")
            # Plot the ROC curve for the current model
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")

        new_row = pd.DataFrame([{
            'Dataset': dataset_name,
            'Model': name,
            'Accuracy': acc,
            'Recall': recall,
            'F1 Score': f1,
            'Precision': precision,
            'AUC': roc_auc
        }])
        results_df = pd.concat([results_df, new_row], ignore_index=True)
    # Plot the random classifier line
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")

    # Plot settings
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Multiple Models")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    plt.savefig(f'./results/fig/exp3_platforms_roc_{dataset_name}.pdf')
    plt.close()
    return results_df

results_dfs = []
for data_spl_file in sorted(data_spl_files):
    print(f'test {data_spl_file}')
    val_loader = dataloader.test_dataloader_cell_gene(args,data_spl_file)
    for idx,loader_idx in enumerate(val_loader):
        results_df = plot_multiple_roc(val_loader[loader_idx], loader_idx)
        results_dfs.append(results_df)

results = pd.concat(results_dfs, ignore_index=True)
results.to_csv(f'./results/exp3_platforms_results.csv', index=False)



