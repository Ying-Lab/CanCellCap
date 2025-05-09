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

args.ckp = os.path.join(args.checkpoint, 'my_model.pt')
args.HVG_list = torch.load(args.ckp)['HVG_list']
args.gene_num = len(args.HVG_list)
args_utils.set_random_seed(args.seed)
args.label_str = ['label', 'tissue_label', 'cancer_label']
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

data_spl_file = args.test_file

args.num_tissue = len(TISCH1_tissue_map.keys())
args.num_cancer_type = len(TISCH1_cancer_type_map.keys())

results_path = args.output

def CanCell_Cap():
    CanCell_Cap = CanCellCap.CanCellCap(args)
    CanCell_Cap.load_state_dict(torch.load(args.ckp)['model_dict'])
    CanCell_Cap.cuda()
    CanCell_Cap.eval()
    return CanCell_Cap

print(f'test {data_spl_file}')
if args.modin:
    test_loader = dataloader.test_dataloader_mpb(args,data_spl_file)
else:
    test_loader = dataloader.test_dataloader(args,data_spl_file)


cancell_cap_model = CanCell_Cap()
pred_pros = []
cell_names = []
pred_labels = []
with torch.no_grad():
    for data in test_loader:
        x = data[0].cuda().float()
        logit = cancell_cap_model.infer(x)
        cell_names.extend(data[4])
        pred_pros.extend(logit.softmax(1).cpu().numpy()) 
        pred_labels.extend(logit.argmax(1).cpu().numpy())
rows = []
pred = []

for cell_name, pred_pro, pred_label in zip(cell_names, pred_pros, pred_labels):
    rows.append([cell_name, pred_pro[0], pred_pro[1], pred_label])

dataset_name = os.path.basename(data_spl_file).split('.')[0]
df = pd.DataFrame(rows, columns=['cell_name', 'normal_probability', 'cancer_probability', 'label'])
df.to_csv(f"{results_path}{dataset_name}_CanCellCap_result.csv", index=False)