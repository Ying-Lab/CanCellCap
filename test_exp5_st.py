# %%
import torch
import pandas as pd 
from admoe_git.models import CanCellCap
from utils import opt_utils,args_utils 

import glob
import os
from admoe_git.data_loaders import dataloader as domian_loaders
import matplotlib.pyplot as plt
import seaborn as sns

from TISCH import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'
args = args_utils.infer_args()

args.ckp = os.path.join(args.output, 'my_model.pt')

args.HVG_list = torch.load(args.ckp)['HVG_list']
args.gene_num = len(args.HVG_list)
args_utils.set_random_seed(args.seed)
args.label_str = ['label', 'tissue_label', 'cancer_label']
args.val_dir = './data/exp5_st/'#colorectal 
SPL_PATH = args.val_dir
data_spl_files = []
if data_spl_files==[]:
    data_spl_files = glob.glob(os.path.join(SPL_PATH,'*.parquet'))

args.num_tissue = len(TISCH1_tissue_map.keys())
args.num_cancer_type = len(TISCH1_cancer_type_map.keys())

results_path = './results/exp5_st/'

def CanCell_Cap():
    algorithm = CanCellCap.CanCellCap(args)
    algorithm.load_state_dict(torch.load(args.ckp)['model_dict'])
    print(f"loading {args.ckp}")
    algorithm.cuda()
    algorithm.eval()
    return algorithm


def plot_multiple_roc(loader, dataset_name="", cell_names=""):

    cancell_cap_model = CanCell_Cap()

    models = {'CanCell_Cap':cancell_cap_model}

    for name in models.keys():
        results_file = f'{results_path}{dataset_name}_{name}_result.csv'
        
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
        for i, (cell_name, label, binary_pred) in enumerate(zip(cell_names, all_labels, all_binary_preds)):

            pred_label = "Cancer" if binary_pred == 1 else "Normal"  # Map binary predictions to labels
            rows.append([cell_name, pred_label])

        # Create DataFrame
        df = pd.DataFrame(rows, columns=['Barcode', 'pred_label'])
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
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(all_labels, all_binary_preds)

        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_normalized = cm / row_sums

        ax = sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues', cbar=False, square=True, xticklabels=["Normal", "Cancer"], yticklabels=["Normal", "Cancer"], annot_kws={"size": 16})

        plt.title("Confusion Matrix for Cancer Spot Identification in ST", fontsize=14)
        plt.xlabel("CanCellCap Predictions", fontsize=14)
        plt.ylabel("Pathologist Annotations", fontsize=14)
        plt.savefig("./results/fig/exp5_ST_confusion_matrix.pdf")
        plt.close()


for data_spl_file in sorted(data_spl_files):
    print(f'test {data_spl_file}')
    val_loader, cell_names = domian_loaders.val_domian_loader_st(args,data_spl_file,shuffle_state=False)
    for idx,loader_idx in enumerate(val_loader):
        plot_multiple_roc(val_loader[loader_idx], loader_idx, cell_names)


