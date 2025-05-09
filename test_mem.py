# %%
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
import time
import tracemalloc
plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['font.family'] = 'Times New Roman'

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
tracemalloc.start()

args = args_utils.infer_args()

args.ckp = os.path.join(args.checkpoint, 'my_model.pt')
args.HVG_list = torch.load(args.ckp)['HVG_list']
args.gene_num = len(args.HVG_list)
args_utils.set_random_seed(args.seed)
args.label_str = ['label', 'tissue_label', 'cancer_label']
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

time_mode=args.time_mode

data_spl_files = [args.test_file]

args.num_tissue = len(TISCH1_tissue_map.keys())
args.num_cancer_type = len(TISCH1_cancer_type_map.keys())

results_path = './results/time/'

def CanCell_Cap():
    CanCell_Cap = CanCellCap.CanCellCap(args)
    CanCell_Cap.load_state_dict(torch.load(args.ckp)['model_dict'])
    CanCell_Cap.cuda()
    CanCell_Cap.eval()
    return CanCell_Cap

def Cancer_Finder():
    algorithm = Cancer_Finder_model.VREx(args)
    algorithm.load_state_dict(torch.load('./ckpt/cancer_finder.pt')['model_dict'])
    algorithm.cuda()
    algorithm.eval()
    return algorithm

CanCell_Cap_model= CanCell_Cap()

models = {'CanCell_Cap':CanCell_Cap_model}

def plot_multiple_roc(model, loader, dataset_name=""):
    plt.figure(figsize=(10, 8))

    all_labels = []
    all_preds = []
    all_binary_preds = []
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            logit = model.infer(x)
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

    return df



for data_spl_file in sorted(data_spl_files):
    print(f'Testing {data_spl_file}...')
    for model_name in models.keys():
        model = models[model_name]

        if args.modin:
            test_loader = dataloader.test_dataloader_mpd(args, data_spl_file)
        else:
            test_loader = dataloader.test_dataloader(args, data_spl_file)

        # Run inference + ROC computation
        dataset_name = os.path.basename(data_spl_file).split('.')[0]
        results_df = plot_multiple_roc(model, test_loader, dataset_name)

        end_time = time.time()
        current_mem, peak_cpu_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if torch.cuda.is_available():
            peak_gpu_mem = torch.cuda.max_memory_allocated()  / 1024 / 1024
        else:
            peak_gpu_mem = 0.0

        peak_cpu_mem_MB = peak_cpu_mem / 1024 / 1024
        total_mem_MB = peak_cpu_mem_MB + peak_gpu_mem

        print(f"Model: {model_name}, Dataset: {dataset_name}")

        print(f"  🧠 CPU Memory Peak: {peak_cpu_mem_MB:.2f} MB")
        print(f"  💻 GPU Memory Peak: {peak_gpu_mem:.2f} MB")
        print(f"  📦 Total Memory Used: {total_mem_MB:.2f} MB")