import os
import collections
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import dgl
import matplotlib.pyplot as plt
import networkx as nx
from random import shuffle
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from transform import reverse
from copy import deepcopy


# load batch
ld = lambda ID: th.load('../data/graphs/' + ID + '.pt')

# cascade batch named tuple. Used to create a "batch of graphs"
# for parallel processing from a list of graphs
CBatch = collections.namedtuple('CBatch', ['graph', 'ID', 'X', 'isroot', 'isleaf', 'y', 'emo'])


def class_ratio(gen):
    # calculate ratio of negative to positive examples for loss weight parameter
    counts = collections.Counter(th.cat([b.y for b in gen]).flatten().tolist())
    return counts[0.] / counts[1.]


def perturbate(x, m, cov, keep=0.90):
    # perturbate root feature
    x = keep * x + (1 - keep) * th.Tensor(np.random.multivariate_normal(m, cov))
    return x


def cascade_batcher(dev):
    def batcher_dev(batch):
        """
        convert list of cascade ids, trees, labels to 
        cascade batch for parallel processing
        """
        ids, trees, labels, emos = zip(*batch)
        batch_trees = dgl.batch(trees)
        cb = CBatch(graph=batch_trees,
                    ID=th.cat(ids), 
                    X=batch_trees.ndata['X'].to(dev),
                    isroot=batch_trees.ndata['isroot'].to(dev),
                    isleaf=batch_trees.ndata['isleaf'].to(dev),
                    y=th.cat(labels).to(dev),
                    emo=th.cat(emos).to(dev))
        return cb
    return batcher_dev


def plot_tree(g, figsize=(8, 8), with_labels=False):
    # this plot requires pygraphviz package
    plt.figure(figsize=figsize)
    pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
    nx.draw(g, pos, with_labels=with_labels, node_size=10,
            node_color='red', arrowsize=4)
    plt.show()


def set_device(_cuda, gpu_id):
    # if _cuda, set device to gpu:gpu_id, else to cpu
    device = 'cpu'
    if _cuda and th.cuda.is_available() and int(gpu_id) < th.cuda.device_count():
        device = 'cuda:' + gpu_id        
    print(device)    
    return th.device(device)


def calc_metrics(ys, y_hats, to_calc, device, criterion=False):
    """
    calculate metrics of cascade lstm
    """    
    metrics = {}    
    sig = nn.Sigmoid().to(device)
    y = th.cat(ys, dim=0)
    # y_hat : output of last layer without sigm
    y_hat = th.cat(y_hats, dim=0)
    # y_prob : prob of being positive in [0, 1]
    y_prob = sig(y_hat)
    # y_pred : predicted label in {0, 1} i.e. only 0 or 1
    y_pred = th.round(y_prob).type(th.FloatTensor).to(device)

    if 'loss' in to_calc and criterion:
        metrics['loss'] = criterion(y_hat, y).item()
    
    if 'acc' in to_calc:
        eq = th.eq(y, y_pred)
        metrics['acc'] = float(th.sum(eq)) / len(y)
        
    if 'auc' in to_calc:
        metrics['auc'] = roc_auc_score(y.cpu(), y_prob.cpu())
    
    if 'precision' in to_calc:
        metrics['precision'] = precision_score(y.cpu(), y_pred.cpu(), average='weighted')

    if 'recall' in to_calc:
        metrics['recall'] = recall_score(y.cpu(), y_pred.cpu(), average='weighted')
        
    if 'f1' in to_calc:
        metrics['f1'] = f1_score(y.cpu(), y_pred.cpu(), average='weighted')
        
    return metrics


def init_net(net):
    # initialize weights of cascade lstm
    for name, param in net.named_parameters():
        if 'bias' in name:
            nn.init.constant(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_normal(param)
            
            
to_save = ('h_size', 'lr_tree', 'lr_top', 'decay_tree', 'decay_top', 'p_drop', 'sample',
           'leaf_ins', 'node_reor', 'emo_pert', 'deep', 'bi', 'structureless', 'test', 'variant')


def log_results(experiment_id, args, epoch, metrics, name='logs_final.csv', to_save=to_save):
    """
    log results of cascade lstm
    """
    dest = args.out_dir + name
    # convert args (model parameters) from command line args to dict
    args_dict = vars(args)
    # select args to save in logs
    log_vars = {k: [args_dict[k]] for k in to_save}
    log_vars['experiment_id'] = [experiment_id]
    log_vars['epoch'] = [epoch]
    # add metrics to model parameters to log
    log_vars = {**log_vars, **{k: [v] for (k, v) in metrics.items()}}
    log_vars['bi'], log_vars['deep'] = bool(log_vars['bi']), bool(log_vars['deep']) 
    df = pd.DataFrame.from_dict(log_vars)
    
    # if logs file exists, append to file ; else create it
    if os.path.exists(dest):
        df.to_csv(dest, header=False, index=False, mode='a')
    else:
        df.to_csv(dest, header=True, index=False, mode='w')
  
      
def node_reordering(dg, iters=5):
    """
    Peform node reordering on a graph:
    In : dgl graph, number of iterations
    Out : dgl graph
    Note : CascadeLSTM needs the nodes pointing as re-tweet -> tweet
        so children are 'predecessors' and parent 'successor'
    """
    
    if iters:
    
        g = dg.to_networkx(node_attrs=['X', 'isroot', 'isleaf'])

        for _ in range(iters):
            # candidate nodes should have children and not be root node
            candidates = [n for n in g.nodes if next(g.predecessors(n), False) and n != 0]
            if not candidates: 
                break
            # pick focus node among candidates
            i = np.random.choice(candidates)
            # children of focus nodes
            ks = list(g.predecessors(i))
            parent = list(g.successors(i))[0]
            # siblings should be child of same parent and be also candidate
            siblings = set(g.predecessors(parent)).intersection(set(candidates))
            siblings.discard(i)
            if siblings:
                # pick random sibling
                j = np.random.choice(list(siblings))
                # disconnect children of focus node
                g.remove_edges_from([(k, i) for k in ks])
                # disconnect sibling from parent
                g.remove_edge(j, parent)
                # insert focus node between parent and sibling
                g.add_edge(j, i, id=j - 1)
                # connect children of focus node to parent
                g.add_edges_from([(k, parent, {'id': k - 1}) for k in ks])
        
        dg.from_networkx(g, node_attrs=['X', 'isroot', 'isleaf'])
        
    return dg


def leaf_insertion(dg, iters=5):
    """
    Perform leaf insertion on a graph
    In : dgl graph, number of iterations
    Out : dgl graph
    Note : CascadeLSTM needs the nodes pointing as re-tweet -> tweet
        so children are 'predecessors' and parent 'successor'
    """
            
    if iters:
                
        for _ in range(iters):
            
            n = dg.number_of_nodes()
            # candidate nodes should not have children
            candidates = [i for i, l in zip(dg.nodes(), dg.ndata['isleaf']) if l]
            # pick focus node among candidates
            i = np.random.choice(candidates)
            # pick w in [0,1]
            w = np.random.random()
            x = dg.ndata['X'][i, :]
            # create children nodes with encoding that sum to encoding of parent
            x1, x2 = w * x, (1 - w) * x
            dg.add_nodes(2, {'X': th.cat([x1, x2]).reshape(2, -1),
                             'isleaf': th.tensor([1, 1]).type(th.float32),
                             'isroot': th.tensor([0, 0]).type(th.float32)})
            # connect children nodes to parent
            dg.add_edges([n, n + 1], [i, i])
            dg.ndata['isleaf'][i] = 0   
        
    return dg
