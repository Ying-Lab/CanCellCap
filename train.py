from models import CanCellCap as model
from utils import opt_utils,args_utils 
import glob
import os
from data_loaders import dataloader
import json
from tqdm import tqdm
from itertools import cycle, zip_longest
from TISCH import *

args = args_utils.get_args()
os.makedirs(args.checkpoint,exist_ok=True)
args.num_tissue = len(TISCH1_tissue_map.keys())
args.num_cancer_type = len(TISCH1_cancer_type_map.keys())
args.label_str = ['label', 'tissue_label', 'cancer_label']
print(args)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

args_utils.set_random_seed(args.seed)

with open(f'./intersection_genome_names.json', 'r') as f:
    args.HVG_list = json.load(f)
    args.gene_num = len(args.HVG_list)

SPL_PATH = args.train_dir
args.data_spl_files = glob.glob(os.path.join(SPL_PATH,'*.parquet'))

train_loaders = dataloader.train_dataloaders(args)
val_loaders = dataloader.val_dataloaders(args)

f_loss_io = open( os.path.join(args.checkpoint,f'{args.logs_name}_loss.txt'),'w')
f_val_io = open( os.path.join(args.checkpoint,f'{args.logs_name}_val.txt'),'w')
[print(_,file=f_val_io,end='\t') if idx!=len(val_loaders)-1 else print(_,file=f_val_io,end='\n') for idx, _ in enumerate(val_loaders) ]

if args.model_name == 'common_moe_mask':
    CanCellCap_model = model.CanCellCap(args)
elif args.model_name == 'moe_mask':
    CanCellCap_model = model.moe_mask(args)
elif args.model_name == 'common_mask':
    CanCellCap_model = model.common_mask(args)
elif args.model_name == 'common':
    CanCellCap_model = model.common(args)
elif args.model_name == 'common_moe':
    CanCellCap_model = model.common_moe(args)

lengths = [len(loader) for loader in train_loaders]
max_length_index = lengths.index(max(lengths))
iterators = [cycle(loader) if i != max_length_index else loader for i, loader in enumerate(train_loaders)]

for epoch in range(args.max_epoch):
    train_minibatches_iterator = zip_longest(*iterators)
    for single_train_minibatches in tqdm(train_minibatches_iterator): 

        if any(mb is None for mb in single_train_minibatches):
            break

        CanCellCap_model.train()

        CanCellCap_model.cuda()
        minibatches_device = [(data) for data in single_train_minibatches]      

        opt = opt_utils.get_optimizer(CanCellCap_model, args)
        sch = opt_utils.get_scheduler(opt, args)
        # back-propagation
        step_vals = CanCellCap_model.update(minibatches_device, opt, sch)
        print(step_vals,file=f_loss_io) 

    CanCellCap_model.eval()

    for idx,loader_idx in enumerate(val_loaders):
        acc = opt_utils.accuracy(CanCellCap_model,val_loaders[loader_idx])
        if idx!=len(val_loaders)-1:
            print (f'val acc: {acc:.4f}',file=f_val_io,end='\t')
        else :
            print (f'val acc: {acc:.4f}',file=f_val_io,end='\n')
        print(f'{acc:.4f}',end='\t')
    f_val_io.flush()   
    print(f'epoch={epoch}',end='\n')    
    print(step_vals)
    
opt_utils.save_checkpoint(f'my_model.pt', CanCellCap_model, args)
f_val_io.close()
f_loss_io.close()

