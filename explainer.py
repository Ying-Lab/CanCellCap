import torch
import pandas as pd
import numpy as np
import os
import glob
import shap
import matplotlib.pyplot as plt
import gseapy as gp
import math
import matplotlib.colors as mcolors
from captum.attr import IntegratedGradients, DeepLift, GradientShap
from models import CanCellCap
from utils import opt_utils, args_utils
from TISCH import *
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'
def init_model(args, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    HVG_list = checkpoint['HVG_list']
    args.HVG_list = HVG_list
    args.gene_num = len(HVG_list)
    args.num_tissue = len(TISCH1_tissue_map.keys())
    args.num_cancer_type = len(TISCH1_cancer_type_map.keys())
    args.label_str = ['label', 'tissue_label', 'cancer_label']
    args_utils.set_random_seed(args.seed)

    model = CanCellCap.CanCellCap(args)
    model.load_state_dict(checkpoint['model_dict'])
    model.cuda()
    model.eval()
    return model, HVG_list

def dataset_external(args, val_dataset, HVG_list):
    gene_list = pd.DataFrame(index=HVG_list)
    gene_list.index = gene_list.index.append(pd.Index(args.label_str))
    raw_df = pd.read_parquet(val_dataset)
    raw_df = raw_df.reindex(gene_list.index).fillna(0)
    raw_df = raw_df[~raw_df.index.duplicated()]
    raw_df = raw_df.loc[gene_list.index]
    raw_df = raw_df.T
    X = torch.from_numpy(raw_df.iloc[:, :args.gene_num].to_numpy()).float().cuda()
    Y = torch.from_numpy(raw_df.loc[:, 'label'].to_numpy()).float().cuda()
    return X, Y

def get_explainer(method, model):
    if method == 'ig':
        return IntegratedGradients(model)
    elif method == 'deeplift':
        return DeepLift(model)
    elif method == 'gradientshap':
        return GradientShap(model)
    else:
        raise ValueError(f"Unsupported method: {method}")

def get_top_genes(args, model, explainer_name, HVG_list, sampled_data_dict, top_n=300):
    top_genes_dict = {}
    for dataset_name, X_sample in sampled_data_dict.items():
        print(f"Processing {dataset_name} with {explainer_name}")
        if explainer_name == 'shap':
            background = X_sample[:-X_sample.shape[0] // 2]
            explain_data = X_sample[-X_sample.shape[0] // 2:]
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(explain_data, check_additivity=False)
            attr = np.array(shap_values)
            attr_target = attr[:, :, 1]
            mean_attr = np.mean(attr_target, axis=0)
        else:
            explainer = get_explainer(explainer_name, model)
            if explainer_name == 'gradientshap':
                baseline = X_sample[:10]
                attributions = explainer.attribute(X_sample, baselines=baseline, target=1)
            else:
                attributions = explainer.attribute(X_sample, target=1)
            mean_attr = attributions.mean(dim=0).detach().cpu().numpy()

        mean_attr_abs = np.abs(mean_attr)
        sorted_idx = np.argsort(mean_attr_abs)[::-1]
        sorted_genes = np.array(HVG_list)[sorted_idx]
        print(f'top 10: {sorted_genes[:10]}')
        top_genes_dict[dataset_name] = set(sorted_genes[:top_n])
    return top_genes_dict

def collect_mean_attr(args, model, explainer_name, HVG_list, sampled_data_dict):
    mean_attr_dict = {}
    for dataset_name, X_sample in sampled_data_dict.items():
        if explainer_name == 'shap':
            background = X_sample[:-X_sample.shape[0] // 2]
            explain_data = X_sample[-X_sample.shape[0] // 2:]
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(explain_data, check_additivity=False)
            attr = np.array(shap_values)
            attr_target = attr[:, :, 1]
            mean_attr = np.mean(attr_target, axis=0)
        else:
            explainer = get_explainer(explainer_name, model)
            if explainer_name == 'gradientshap':
                baseline = X_sample[:10]
                attributions = explainer.attribute(X_sample, baselines=baseline, target=1)
            else:
                attributions = explainer.attribute(X_sample, target=1)
            mean_attr = attributions.mean(dim=0).detach().cpu().numpy()
        mean_attr_dict[dataset_name] = mean_attr
    return mean_attr_dict

def plot_top_features_grid(mean_attr_dict, HVG_list, explainer_name):

    os.makedirs('./fig', exist_ok=True)
    num_datasets = len(mean_attr_dict)
    cols = 4
    rows = math.ceil(num_datasets / cols)
    cmap = mcolors.LinearSegmentedColormap.from_list("red_blue", ["red", "blue"])
    colors = cmap(np.linspace(1, 0, 10))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, (dataset, mean_attr) in enumerate(mean_attr_dict.items()):
        r, c = divmod(i, cols)
        ax = axes[r][c]

        mean_attr_abs = np.abs(mean_attr)
        sorted_idx = np.argsort(mean_attr_abs)[::-1][:10]
        sorted_genes = np.array(HVG_list)[sorted_idx]
        sorted_values = mean_attr_abs[sorted_idx]

        ax.barh(range(10), sorted_values[::-1], color=colors)
        ax.set_yticks(range(10))
        ax.set_yticklabels(sorted_genes[::-1])
        ax.set_title(f'{dataset}')

    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j // cols][j % cols])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = f'./fig/top_10_gene_{explainer_name}_all.pdf'
    plt.savefig(output_path, format='pdf')
    plt.close()
    print(f"Combined feature importance plot saved to: {output_path}")

if __name__ == "__main__":
    args = args_utils.infer_args()
    args.ckp = os.path.join(args.checkpoint, 'my_model.pt')
    model, HVG_list = init_model(args, args.ckp)

    data_path = './data/train_data/'
    data_files = glob.glob(os.path.join(data_path, '*.parquet'))

    sampled_data_dict = {}
    for data_file in sorted(data_files):
        dataset_name = os.path.basename(data_file).split('.')[0]
        X_test, y_test = dataset_external(args, data_file, HVG_list)
        sample_size = min(200, y_test.size(0))
        X_sample = X_test[np.random.choice(X_test.shape[0], sample_size, replace=False)]
        sampled_data_dict[dataset_name] = X_sample

    explain_tool1 = "shap"
    explain_tool2 = "gradientshap"
    # explain_tool1 = "ig"
    # explain_tool2 = "deeplift"

    tool1_top_genes = get_top_genes(args, model, explain_tool1, HVG_list, sampled_data_dict)
    tool2_top_genes = get_top_genes(args, model, explain_tool2, HVG_list, sampled_data_dict)

    print("Dataset\tOverlap Count")
    for dataset in tool1_top_genes:
        if dataset in tool2_top_genes:
            overlap = list(tool1_top_genes[dataset].intersection(tool2_top_genes[dataset]))
            print(f"{dataset}\t{len(overlap)}")

            enr = gp.enrichr(
                gene_list=overlap,
                gene_sets=['KEGG_2021_Human'],
                organism='Human',
                outdir='enrichr_result',
                cutoff=0.5
            )

            enr.results['enrichment'] = enr.results['Odds Ratio']
            enr.results['count'] = enr.results['Overlap'].str.split('/').str[0].astype(int)
            out_plot = enr.results.loc[:, ['Term', 'enrichment', 'P-value', 'count']]
            out_plot_sorted = out_plot.sort_values(by='count', ascending=False)
            print(out_plot_sorted.head(10).to_csv(index=False, sep='\t'))
            out_plot_sorted.head(10).to_csv(f'./fig/{dataset}_KEGG_{explain_tool1}_{explain_tool2}.tsv', index=False, sep='\t')

    tool1_attr_dict = collect_mean_attr(args, model, explain_tool1, HVG_list, sampled_data_dict)
    tool2_attr_dict = collect_mean_attr(args, model, explain_tool2, HVG_list, sampled_data_dict)
    plot_top_features_grid(tool1_attr_dict, HVG_list, explain_tool1)
    plot_top_features_grid(tool2_attr_dict, HVG_list, explain_tool2)
