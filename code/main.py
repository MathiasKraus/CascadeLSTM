import os, sys, warnings

try : os.chdir(os.path.dirname(sys.argv[0]))
except : pass

sys.path.append('../code/')

import numpy as np
import torch as th
import dgl
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from random import shuffle
import torch.nn as nnz
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from utils import *
from collections import Counter
from klasses import CascadeData
from trees import DeepTreeLSTM

if not sys.warnoptions:
    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--g_dir', dest = 'graphs_dir', default = '../data/graphs/')
parser.add_argument('--x_size', dest = 'x_size', default = 12, type = int)
parser.add_argument('--h_size', dest = 'h_size', default = 3, type = int)
parser.add_argument('--nc', dest = 'num_classes', default = 1, type = int)
parser.add_argument('--sample', dest = 'sample', action='store_true')
parser.add_argument('--bsize', dest = 'batch_size', default = 25, type = int)
parser.add_argument('--pd', dest = 'p_drop', default = 0.1, type = float)
parser.add_argument('--epochs', dest = 'epochs', default = 50, type = int)
parser.add_argument('--lr_tree', dest = 'lr_tree', default = 0.01, type = float)
parser.add_argument('--lr_top', dest = 'lr_top', default = 0.01, type = float)
parser.add_argument('--decay_tree', dest = 'decay_tree', default = 0.003, type = float)
parser.add_argument('--decay_top', dest = 'decay_top', default = 0.006, type = float)
parser.add_argument('--out_dir', dest = 'out_dir', default = '../results/')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu', dest = 'gpu', type = str, default = "")
parser.add_argument('--leaf_ins', action='store_true', default = False)
parser.add_argument('--node_reor', action='store_true', default = False)
parser.add_argument('--emo_pert', action='store_true', default = False)
parser.add_argument('--verbose', action='store_true', default = False)
parser.add_argument('--save_ids', action='store_true', default = False)
parser.add_argument('--deep', default = 1, type=int)
parser.add_argument('--bi', default = 1, type=int)
parser.add_argument('--variant', type = str, default = '')
parser.add_argument('--structureless', action='store_true', default = False)
parser.add_argument('--test', action='store_true', default = False)



args = parser.parse_args()
variant = args.variant

experiment_id = datetime.now().strftime("%m_%d_%Y__%H_%M_%S__%f") 

verb = args.verbose

if args.cuda:
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
    if verb: print("There are %d devices available" % th.cuda.device_count())
    
device = set_device(args.cuda, args.gpu)

sig = nn.Sigmoid().to(device)

def learn(model, batch, device, h_size):
    n = batch.graph.number_of_nodes()
    h = th.zeros((n, h_size)).to(device)
    c = th.zeros((n, h_size)).to(device)
    y_hat = model(batch, h, c)
    return batch.y.to(device), y_hat

ids = np.array([ID.split('.')[0] for ID in os.listdir(args.graphs_dir) if '_' not in ID])

if not args.test:
    split_ratio = 0.85
    split = int(len(ids) * split_ratio)
    shuffle(ids)
    train_ids, test_ids = ids[:split], ids[split:]
else:
    train_ids = ids
    test_ids = np.unique([ID.split('_')[0] for ID in os.listdir(args.graphs_dir) if 'test' in ID])


train_set = CascadeData(train_ids, args.sample, args.leaf_ins, args.node_reor, args.emo_pert, variant=args.variant, structureless=args.structureless)
test_set = CascadeData(test_ids, variant=args.variant, structureless=args.structureless, test = args.test)

train_generator = DataLoader(train_set, collate_fn=cascade_batcher(device), batch_size= args.batch_size)
test_generator = DataLoader(test_set, collate_fn=cascade_batcher(device), batch_size= args.batch_size)

r_train = class_ratio(train_generator)
r_test = class_ratio(test_generator)


h_size = args.h_size
x_size = args.x_size
if args.structureless: 
    x_size = x_size - 2
emo_size = 8
num_classes = args.num_classes
top_sizes = (16, 16)
p_drop = args.p_drop
epochs = args.epochs
lr_tree = args.lr_tree
decay_tree = args.decay_tree
lr_top = args.lr_top
decay_top = args.decay_top
bi = bool(args.bi)
deep = bool(args.deep)

params_path = args.out_dir + 'model_' + experiment_id + '.pt'
preds_path = args.out_dir + 'preds_' + experiment_id + '.csv'


deep_tree = DeepTreeLSTM(x_size, h_size, num_classes, emo_size, top_sizes, p_drop, bi=bool(bi), deep=bool(deep), logit=False)

if args.cuda and th.cuda.is_available():
	deep_tree.to(device)

init_net(deep_tree)

criterion = nn.BCEWithLogitsLoss(pos_weight = th.Tensor([r_train])).to(device)
crit_eval = nn.BCEWithLogitsLoss(pos_weight = th.Tensor([r_test])).to(device)
optimizer_tree = optim.Adam(deep_tree.bottom_net.parameters(), lr = lr_tree, weight_decay = decay_tree)
optimizer_top = optim.Adam(deep_tree.top_net.parameters(), lr = lr_top, weight_decay = decay_top)
scheduler_tree = optim.lr_scheduler.StepLR(optimizer_tree, step_size=5, gamma=0.9)
scheduler_top = optim.lr_scheduler.StepLR(optimizer_top, step_size=5, gamma=0.9)


best_state, best_metrics, best_epoch = None, {'auc':0.}, 0.

print("Started experiment  " + experiment_id)

for epoch in range(epochs):
    
    if best_epoch + 8 < epoch:
        break
    
    deep_tree.train()
    
    ys, y_hats = [], []
    
    for step, batch in enumerate(train_generator):
        y, y_hat = learn(deep_tree, batch, device, h_size)
        optimizer_tree.zero_grad()
        optimizer_top.zero_grad()
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer_tree.step()
        optimizer_top.step()
        ys.append(y.detach())
        y_hats.append(y_hat.detach())
        if verb: 
            print("#", end="", flush=True)
    
    scheduler_tree.step()
    scheduler_top.step()
    if verb: 
        print("")
        
    deep_tree.eval()
    
    with th.no_grad():
        
        tr_metrics = calc_metrics(ys, y_hats, ['loss', 'acc', 'auc'], device, criterion)
    
        ys, y_hats = [], []
        
        test_ids = []
        
        for step, batch in enumerate(test_generator):
            y, y_hat = learn(deep_tree, batch, device, h_size)
            ys.append(y.detach())
            y_hats.append(y_hat.detach())
            test_ids.append(batch.ID)

        te_metrics = calc_metrics(ys, y_hats, ['loss', 'acc', 'auc', 'precision', 'recall', 'f1'], device, criterion)
            
                
    if verb: print("Epoch {:03d} | "
                   "Train Loss {:.3f} | Train Acc {:.3f} | Train AUC {:.3f} | "
                   "Test Loss {:.3f} | Test Acc {:.3f} | Test AUC {:.3f}".format(epoch, 
    tr_metrics['loss'], tr_metrics['acc'], tr_metrics['auc'],
    te_metrics['loss'], te_metrics['acc'], te_metrics['auc']))
    
    if te_metrics['auc'] > best_metrics['auc'] :
        best_metrics = te_metrics
        best_state = deep_tree.state_dict()
        best_epoch = epoch
        th.save(best_state, params_path)
        ys = th.cat(ys, dim = 0).flatten().tolist()
        y_hats = sig(th.cat(y_hats, dim = 0)).flatten().tolist()
        test_ids = th.cat(test_ids, dim = 0).flatten().tolist()
        df_res = pd.DataFrame({'id':test_ids, 'y':ys, 'y_hat':y_hats})
        df_res.to_csv(preds_path, index=False, header=True)
        print ("Model %s saved with test AUC %.4f | train AUC %.4f at epoch %d" % (experiment_id, te_metrics['auc'], tr_metrics['auc'], epoch))

print ("Experiment %s terminated with test AUC %.4f at epoch %d" % (experiment_id, best_metrics['auc'], epoch))
        
log_results(experiment_id, args, best_epoch, best_metrics)
    