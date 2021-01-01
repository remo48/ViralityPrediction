import os, sys
import time
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

# load batch
def ld(ID, graphs_dir):
    return th.load(graphs_dir + ID + '.pt')


# cascade batch named tuple. Used to create a "batch of graphs"
# for parallel processing from a list of graphs
CBatch = collections.namedtuple('CBatch', ['graph', 'ID', 'X', 'isroot', 'isleaf', 'y', 'emo'])


class Logger:
    
    def __init__(self, verbose=1):
        super().__init__()

        self.verbose = verbose
        self.indent = 4
        self.time_width = 10
        self.terminal_width = os.get_terminal_size().columns
        self.max_length = self.terminal_width - self.time_width - 4

    def make_title(self, title):
        if self.verbose > 0:
            sys.stdout.write(title+'\n')
            sys.stdout.flush()

    def step_start(self, str):
        if self.verbose > 1:
            self.out = self.indent*' ' + str
            self.start = time.time()
            sys.stdout.write(self.out)
            sys.stdout.flush()

    def step_end(self, info=''):
        if self.verbose > 1:
            elapsed = time.time() - self.start

            out = self.out + ' ' + info

            if elapsed > 60:
                time_format = '%d:%02d' % (elapsed // 60, elapsed % 60)
            else:
                time_format = '%.2fs' % elapsed
            pass

            sys.stdout.write('\r')
            sys.stdout.write('{:{length}}{:>{time_length}}\n'.format(out, time_format, length=self.max_length, time_length=self.time_width))
            sys.stdout.flush()


def get_class_weights(dataloader):
    target_list = []
    counts = []

    for _,_,t,_ in dataloader:
        target_list.append(t)
    target_list = th.tensor(target_list).long()

    for i in target_list.unique():
        counts.append((target_list == i).sum())

    class_weights = 1./th.tensor(counts, dtype=th.float)
    return class_weights, class_weights[target_list]



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

def init_net(net):
    for name, param in net.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_normal_(param)


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