
import torch
import pandas as pd 
from models import CanCellCap
from models import Cancer_Finder_model
from utils import args_utils 
import glob
import os
from data_loaders import dataloader
import matplotlib.pyplot as plt
from TISCH import *
import torch
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score, precision_score, recall_score
import numpy as np
from sklearn.neighbors import NearestNeighbors
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'

def calculate_connectivities_from_embedding(embedding, n_neighbors):
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(embedding)
    distances, indices = knn.kneighbors(embedding)
    n_cells = embedding.shape[0]
    connectivity_matrix = np.zeros((n_cells, n_cells))
    for i, neighbors in enumerate(indices):
        for j, neighbor in enumerate(neighbors):
            connectivity_matrix[i, neighbor] = np.exp(-distances[i, j]/4)  # 用高斯核加权
            # connectivity_matrix[i, neighbor] = 1 / n_neighbors
    connectivity_matrix /= connectivity_matrix.sum(axis=1, keepdims=True)
    return connectivity_matrix

def propagate_labels(cell_preds, connectivity_matrix, n_iter=1, certainty_threshold=0.4):
    pred_proba_class_1 = cell_preds[:, 1].cpu().numpy()  # Class 1 的概率
    pred_proba_class_0 = cell_preds[:, 0].cpu().numpy()  # Class 0 的概率
    scores = cell_preds.max(dim=1)[0].cpu().numpy()  # 置信度 = max(p_0, p_1)
    pred_proba = pd.DataFrame({
        "Class_0_Prob": pred_proba_class_0,
        "Class_1_Prob": pred_proba_class_1
    })
    certainty_info = pd.Series(scores)
    final_pred_proba = pred_proba.copy() 

    for i in range(n_iter):
        certainty_threshold_pct = certainty_threshold #* np.exp(-0.3 * i)
        # certain_cells = certainty_info > np.quantile(certainty_info, certainty_threshold_pct)
        updated_pred_proba = np.dot(connectivity_matrix, final_pred_proba.values)
        updated_pred_proba = updated_pred_proba.astype(np.float32)
        # final_pred_proba.loc[~certain_cells] = updated_pred_proba[~certain_cells]
        final_pred_proba.loc[:] = updated_pred_proba[:]
        final_pred_proba = final_pred_proba.clip(1e-6, 1 - 1e-6)
    final_labels = (final_pred_proba["Class_1_Prob"] > 0.5).astype(int)  # 基于 Class_1 的概率
    final_pred_tensor = torch.tensor(final_pred_proba.values, dtype=torch.float32)  # (N, 2) tensor
    final_labels_tensor = torch.tensor(final_labels.values, dtype=torch.int64)  # (N,) tensor
    # breakpoint()
    return final_labels_tensor, final_pred_tensor


args = args_utils.infer_args()

args.ckp = os.path.join(args.checkpoint, 'my_model.pt')
args.HVG_list = torch.load(args.ckp)['HVG_list']
args.gene_num = len(args.HVG_list)
args_utils.set_random_seed(args.seed)
args.label_str = ['label', 'tissue_label', 'cancer_label']

SPL_PATH = "/root/CanCellCap/data/new_cancer_type"

data_spl_files = []
if data_spl_files==[]:
    data_spl_files = glob.glob(os.path.join(SPL_PATH,'*data.csv')) + glob.glob(os.path.join(SPL_PATH, '*.parquet'))

args.num_tissue = len(TISCH1_tissue_map.keys())
args.num_cancer_type = len(TISCH1_cancer_type_map.keys())

results_path = './results/new_cancer_type/'

def CanCell_Cap():
    CanCell_Cap = CanCellCap.CanCellCap(args)
    CanCell_Cap.load_state_dict(torch.load(args.ckp)['model_dict'])
    print(f"loading {args.ckp}")
    CanCell_Cap.cuda()
    CanCell_Cap.eval()
    return CanCell_Cap

def Cancer_Finder():
    algorithm = Cancer_Finder_model.VREx(args)
    algorithm.load_state_dict(torch.load('/root/ori/new_trainset/my_model.pt')['model_dict'])
    algorithm.cuda()
    algorithm.eval()
    return algorithm


cancell_cap_model = CanCell_Cap()
Cancer_Finder_model=Cancer_Finder()

def plot_multiple_roc(loader, dataset_name=""):
    """
    Plot ROC curves for multiple models.

    Args:
        models (list): List of models to evaluate.
        loader: DataLoader for the dataset.
    """
    results_df = pd.DataFrame(columns=['Dataset', 'Model', 'Accuracy', 'Recall', 'F1 Score', 'Precision', 'AUC'])
    plt.figure(figsize=(10, 8))
    models = {'CanCell_Cap':cancell_cap_model, 'Cancer_Finder':Cancer_Finder_model, 'PreCanCell':'precancell','SCEVAN':'SCEVAN','CopyKAT':'CopyKAT', 'ikarus':'ikarus'}
    pre_result_model = ['Cancer_Finder','PreCanCell','SCEVAN','CopyKAT','ikarus']
    non_roc_model = ['SCEVAN','CopyKAT']
    for name in models.keys():
        results_file = f'{results_path}{dataset_name}_{name}_result.csv'
        if name in pre_result_model:
            # Sample data
            if not os.path.exists(results_file):
                print(f'{results_file} is not exist.')
                continue

            df = pd.read_csv(results_file, index_col=0)
            try:
                all_labels = df['label']
            except:
                all_labels = df['labels']
                df.rename(columns={'labels': 'label'}, inplace=True)
                df.to_csv(results_file)
            y_scores = df['freq_cancer']
            all_binary_preds = df['pred'].fillna(2).astype(int)
            # Calculate accuracy and F1 score
            acc = accuracy_score(all_labels, all_binary_preds)
            all_binary_preds_adj = np.where(all_binary_preds == 2, 1 - all_labels, all_binary_preds)
            recall = recall_score(all_labels, all_binary_preds_adj)
            f1 = f1_score(all_labels, all_binary_preds_adj)
            precision = precision_score(all_labels, all_binary_preds_adj)

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
            cell_embeddings_list = []
            with torch.no_grad():
                for data in loader:

                    x = data[0].cuda().float()
                    y = data[1].cuda().long()

                    logit, x_cat = models[name].infer_embedding(x)
                    cell_embeddings_list.append(x_cat)
                    # Collect predictions and labels
                    all_labels.extend(y.cpu().numpy())
                    all_preds.extend(logit.softmax(1).cpu().numpy())  # Use softmax probabilities
                    all_binary_preds.extend(logit.argmax(1).cpu().numpy())  # Predicted labels for accuracy, F1, precision

            cell_embeddings  = torch.cat(cell_embeddings_list)
            cell_preds = torch.tensor(np.array(all_preds))
            connectivity_matrix = calculate_connectivities_from_embedding(cell_embeddings.cpu(), 10)

            final_pred, final_pred_proba = propagate_labels(cell_preds, connectivity_matrix)

            all_binary_preds, all_preds = final_pred, final_pred_proba
            rows = []
            for i, (pred, label, binary_pred) in enumerate(zip(all_preds, all_labels, all_binary_preds)):
                freq_cancer = pred[1]  # Assuming index 1 is for cancer class
                freq_non_cancer = pred[0]  # Assuming index 0 is for non-cancer class
                pred_label = "Cancer" if binary_pred == 1 else "Normal"  # Map binary predictions to labels
                rows.append([i, freq_cancer, freq_non_cancer, pred_label, binary_pred, label])

            # Create DataFrame
            df = pd.DataFrame(rows, columns=['', 'freq_cancer', 'freq_non_cancer', 'pred_label', 'pred', 'label'])

            # Save to CSV
            df.to_csv(f"{results_path}{dataset_name}_{name}_result.csv", index=False)

            # Calculate evaluation metrics
            acc = accuracy_score(all_labels, all_binary_preds)
            f1 = f1_score(all_labels, all_binary_preds)
            precision = precision_score(all_labels, all_binary_preds)
            recall = recall_score(all_labels, all_binary_preds)

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
    plt.savefig(f'./results/fig/new_cancer_type_roc_{dataset_name}.pdf')
    return results_df

results_dfs = []
for data_spl_file in sorted(data_spl_files):
    print(f'test {data_spl_file}')
    val_loader = dataloader.test_dataloader(args,data_spl_file,log=True)
    for idx,loader_idx in enumerate(val_loader):
        results_df = plot_multiple_roc(val_loader[loader_idx], loader_idx)
        results_dfs.append(results_df)

results = pd.concat(results_dfs, ignore_index=True)
results.to_csv(f'./results/new_cancer_type_results.csv', index=False)

