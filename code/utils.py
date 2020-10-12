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
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, accuracy_score
from copy import deepcopy
import time

from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NeighbourhoodCleaningRule
from imblearn.over_sampling import RandomOverSampler


class Pbar:
    '''
    class to visualize training status
    '''
    def __init__(self, target, epoch, num_epochs ,width=40) -> None:
        '''
        Initialize class.
        In :
            - target: target value of Progress bar
            - epoch: current epoch
            - num_epochs: total number of epochs
            - width: width of progress bar
        '''
        self.target = target
        self.epoch = epoch
        self.num_epochs = num_epochs
        self.width = width

        self._start = time.time()
        self._seen_so_far = 0
        self._metrics = {}

        print(f'Epoch: {self.epoch+1}/{self.num_epochs}')

    def __metrics_str(self, values):
        values_str = ''
        for name, value in values:
            values_str += f'- {name}: {value:.4f} '
        
        return values_str

    def update(self, step, values):
        now = time.time()
        left = (step+1) * self.width // self.target
        right = self.width - left

        finalize = (step+1) >= self.target

        time_per_step = (now - self._start) / (step+1)
        time_format = ''

        if finalize:
            if time_per_step >= 1 or time_per_step == 0:
                time_per_step_format = ' %.0fs/step' % (time_per_step)
            elif time_per_step >= 1e-3:
                time_per_step_format = ' %.0fms/step' % (time_per_step * 1e3)
            else:
                time_per_step_format = ' %.0fus/step' % (time_per_step * 1e6)

            elapsed = now - self._start
            if elapsed > 3600:
                time_format = '%d:%02d:%02d' % (elapsed // 3600,
                                            (elapsed % 3600) // 60, elapsed % 60)
            elif elapsed > 60:
                time_format = '%d:%02d' % (elapsed // 60, elapsed % 60)
            else:
                time_format = '%ds' % elapsed

            time_format += time_per_step_format 

        else:
            eta = (self.target - step) * time_per_step
            if eta > 3600:
                eta_format = '%d:%02d:%02d' % (eta // 3600,
                                            (eta % 3600) // 60, eta % 60)
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta

            time_format = f'ETA: {eta_format}'
        

        metrics_str = ''
        for k, v in values:
            if k not in self._metrics:
                self._metrics[k] = v
            else:
                self._metrics[k] += v

            avg = self._metrics[k] / (step+1)
            metrics_str += f'- {k}: {avg:.4f} '

        print(f'\r{step+1}/{self.target} ',
              '[', '#'*left, ' '*right, '] ',
              f'- {time_format} ', 
              metrics_str, sep='', end='', flush=True)

    def add_val(self, values):
        print(self.__metrics_str(values), end='\n')


# load batch
def ld(ID, graphs_dir):
    return th.load(graphs_dir + ID + '.pt')


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


def set_device(cuda_id):
    device = 'cpu'
    if cuda_id >= 0 and th.cuda.is_available() and int(cuda_id) < th.cuda.device_count():
        device = 'cuda:' + cuda_id        
    return th.device(device)


def calc_metrics(y, y_hat, to_calc, device, criterion=None):
    """
    calculate metrics of cascade lstm
    """    
    metrics = {}    
    sig = nn.Sigmoid().to(device)
    y_prob = sig(y_hat)
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

    if 'rmse' in to_calc:
        metrics['rmse'] = mean_squared_error(y.cpu(), y_hat.cpu())**.5

    if 'mse' in to_calc:
        metrics['mse'] = mean_squared_error(y.cpu(), y_hat.cpu())

    if 'mae' in to_calc:
        metrics['mae'] = mean_absolute_error(y.cpu(), y_hat.cpu())
         
    return metrics


def init_net(net):
    # initialize weights of cascade lstm
    for name, param in net.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_normal_(param)
            
            
to_save = ('h_size', 'lr_tree', 'lr_top', 'decay_tree', 'decay_top', 'p_drop', 'sample',
           'leaf_ins', 'node_reor', 'emo_pert', 'deep', 'bi', 'structureless', 'variant')


def log_results(experiment_id, args, out_dir, epoch, metrics, name='logs_final.csv', to_save=to_save):
    """
    log results of cascade lstm
    """
    dest = out_dir + name
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
    g = None
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
        
    return dgl.from_networkx(g, node_attrs=['X', 'isroot', 'isleaf'])


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


samplers = [RandomOverSampler(sampling_strategy=0.3), TomekLinks(), NeighbourhoodCleaningRule()]

class IterativeSampler():
    """
    Applies a sequence of samplers. If the "return_ids" parameter
    is set to True, also returns the inices in the data frame or
    the matrix of the samples that have been selected. 
    """
    def __init__(self, samplers=samplers, return_ids=True):
        self.samplers = samplers
        self.return_ids = return_ids
        if return_ids:
            self.dict_ids = {}
        
    def fit_sample(self, X, y):

        for sampler in self.samplers:
            X, y = sampler.fit_sample(X, y)
            if self.return_ids:
                new_ids = list(range(len(y)))
                if self.dict_ids:
                    d0 = dict(zip(new_ids, sampler.sample_indices_))
                    d = {k: self.dict_ids[v] for k, v in d0.items()}
                else:
                    d = dict(zip(new_ids, sampler.sample_indices_))
                    
                self.dict_ids = d
                
        if self.return_ids:
            return X, y, self.dict_ids
        else:
            return X, y