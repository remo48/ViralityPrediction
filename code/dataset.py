import torch as th
import dgl
from utils import *
import torch as th
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from random import shuffle


class Cascade:
    """
    class that contains a single cascade. The cascade data structure
    is save and then loaded in ".pt" ~ pytorch format.
    """
    def __init__(self, cascade_id, X, emo, src, dest, isleaf):
        # init the cascade data structure with data

        self.cascade_id = cascade_id
        # node covariates
        self.X = X
        self.size = X.shape[0]
        self.isleaf = isleaf
        self.isroot = th.cat([th.Tensor([1]), th.zeros(self.size - 1)])
        self.emo = emo
        self.edges = {'src': src, 'dest': dest}

    def make_graph(self):
        """
        reconstruct dgl (deep graph library) graph from saved nodes,
        edges and node data
        """

        src, dest = self.edges['src'], self.edges['dest']
        g = dgl.graph((src, dest), num_nodes=self.size)
        g.ndata['X'] = self.X
        g.ndata['isroot'] = self.isroot
        g.ndata['isleaf'] = self.isleaf

        return g
    
    def retrieve_data(self):
        """
        retrieve dgl graph, label and root features from cascade data structure.
        If respective arg is set to True, perform data augmentation
        """
        
        g = self.make_graph()
        e = self.emo

        return (g, e)
    

class CascadeData(Dataset):
    """
    Class that loads all the ids it is initialized with.
    Note that two instances are utilized when learning ; 
    one for trainin and one for validation or testing
    """

    def __init__(self, 
                list_IDs, 
                graphs_dir,
                cascade_size_file,
                categorical=False,
                variant='',
                structureless = False,
                test=False):
        """
        Initialize class.
        In :
            - list_IDs: list of ids to load
            - variant: crop e.g. "1000_tweets"
            - structureless: whether to load the variant without node depth 
            and (log of) number of children  
            - test: load test (rather than train) data
        """

        self.variant = ['', '_' + variant][variant != '']
        self.structureless = ['', '_structureless'][structureless]
        self.test = ['', '_test'][test]

        self.log = {
            'variant': variant,
            'structureless': structureless,
        }

        target = 'cascade_size_log'
        if categorical:
            target = 'category'

        shuffle(list_IDs)

        self.list_IDs = list_IDs

        df_cascade_size = pd.read_csv(cascade_size_file)
        self.cascades = []
        self.y = []
        for ID in list_IDs:
            cascade_id = ID + self.variant + self.structureless + self.test
            self.cascades.append(ld(cascade_id, graphs_dir).retrieve_data())
            y = df_cascade_size.loc[df_cascade_size['cascade_id'] == int(ID), target].values
            if categorical:
                self.y.append(th.tensor(y).long())
            else:
                self.y.append(th.tensor([y], dtype=th.float32))


    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        g, emo = self.cascades[index]
        y = self.y[index]

        return th.Tensor([int(ID)]), g, y, emo