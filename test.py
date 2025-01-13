# %%
import torch
import pandas as pd 
from models import CanCellCap
from utils import args_utils 
import glob
import os
from data_loaders import dataloader
import matplotlib.pyplot as plt
from TISCH import *
import torch
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score, precision_score, recall_score

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'

args = args_utils.infer_args()

args.ckp = os.path.join(args.output, 'my_model.pt')
args.HVG_list = torch.load(args.ckp)['HVG_list']
args.gene_num = len(args.HVG_list)
args_utils.set_random_seed(args.seed)
args.label_str = ['label', 'tissue_label', 'cancer_label']

test_PATH = "./data/"

data_spl_files = []
if data_spl_files==[]:
    data_spl_files = glob.glob(os.path.join(test_PATH,'*.parquet'))

args.num_tissue = len(TISCH1_tissue_map.keys())
args.num_cancer_type = len(TISCH1_cancer_type_map.keys())

results_path = './results/'

def CanCell_Cap():
    CanCell_Cap = CanCellCap.CanCellCap(args)
    CanCell_Cap.load_state_dict(torch.load(args.ckp)['model_dict'])
    CanCell_Cap.cuda()
    CanCell_Cap.eval()
    return CanCell_Cap


def plot_multiple_roc(loader, dataset_name=""):
    plt.figure(figsize=(10, 8))
    cancell_cap_model = CanCell_Cap()

    all_labels = []
    all_preds = []
    all_binary_preds = []
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            logit = cancell_cap_model.predict_2(x)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(logit.softmax(1).cpu().numpy())
            all_binary_preds.extend(logit.argmax(1).cpu().numpy()) 

    rows = []
    for i, (pred, label, binary_pred) in enumerate(zip(all_preds, all_labels, all_binary_preds)):
        freq_cancer = pred[1] 
        freq_non_cancer = pred[0] 
        pred_label = "Cancer" if binary_pred == 1 else "Normal" 
        rows.append([i, freq_cancer, freq_non_cancer, pred_label, binary_pred, label])

    df = pd.DataFrame(rows, columns=['', 'freq_cancer', 'freq_non_cancer', 'pred_label', 'pred', 'label'])

    df.to_csv(f"{results_path}{dataset_name}_CanCellCap_result.csv", index=False)

    acc = accuracy_score(all_labels, all_binary_preds)
    f1 = f1_score(all_labels, all_binary_preds, average='weighted')
    precision = precision_score(all_labels, all_binary_preds, average='weighted')
    recall = recall_score(all_labels, all_binary_preds)

    fpr, tpr, _ = roc_curve(all_labels, [pred[1] for pred in all_preds], pos_label=1)
    roc_auc = auc(fpr, tpr)

    print(f"CanCellCap - Accuracy: {acc:.4f}, recall: {recall:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, AUC: {roc_auc:.4f},")
    plt.plot(fpr, tpr, label=f"CanCellCap (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Multiple Models")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    plt.savefig(f'./results/fig/exp1_external_roc_{dataset_name}.pdf')
    return

for data_spl_file in sorted(data_spl_files):
    print(f'test {data_spl_file}')
    val_loader = dataloader.test_dataloader_gene_cell(args,data_spl_file)
    for idx,loader_idx in enumerate(val_loader):
        plot_multiple_roc(val_loader[loader_idx], loader_idx)

