import torch as th
import dgl
from utils import *
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from random import shuffle


class Cascade:
    """
    class that contains a single cascade. The cascade data structure
    is save and then loaded in ".pt" ~ pytorch format.
    """
    def __init__(self, cascade_id, X, target_vars, emo, src, dest, isleaf):
        # init the cascade data structure with data

        self.cascade_id = cascade_id
        # node covariates
        self.X = X
        self.target_vars = target_vars
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
    
    def retrieve_data(self, target_var, leaf_ins=False, node_reor=False, emo_data=False):
        """
        retrieve dgl graph, label and root features from cascade data structure.
        If respective arg is set to True, perform data augmentation
        TODO: adjust for multiple target variables
        """
        
        g = self.make_graph()
        e = self.emo

        if leaf_ins:
            g = leaf_insertion(g)
        if node_reor:
            g = node_reordering(g)
            
        #if emo_data:
        #    e = perturbate(e, emo_data[self.y.item()]['mean'], emo_data[self.y.item()]['cov'])

        return (g, self.target_vars[target_var], e)
    

class CascadeData(Dataset):
    """
    Class that loads all the ids it is initialized with.
    Note that two instances are utilized when learning ; 
    one for trainin and one for validation or testing
    """

    def __init__(self, list_IDs, target_var, data_dir, settings=None, test=False):
        """
        Initialize class.
        In :
            - list_IDs: list of ids to load
            - sample : whether to apply sampling to ids
            - leaf_ins, node_reor, emo_pert: whether to apply espective data 
          augmentation to loaded cascades
            - variant: crop e.g. "1000_tweets"
            - structureless: whether to load the variant without node depth 
            and (log of) number of children  
            - test: load test (rather than train) data
        """
        variant = settings.get('variant', '')
        structureless = settings.get('structureless', False)
        sample = settings.get('sample', False)
        emo_pert = settings.get('emo_pert', False)

        # prepend underscore to variant to load cascade with name ID_variant
        self.variant = ['', '_' + variant][variant != '']
        self.structureless = ['', '_structureless'][structureless]
        self.test = ['', '_test'][test]
        self.data_dir = data_dir

        'Initialization'
        if sample:
            list_IDs = self.sample(list_IDs)    
        if emo_pert:
            self.emo_data = self.get_emo_data(data_file=self.data_dir + 'grouped' + self.variant + self.test + '.csv')
        else:
            self.emo_data = False

        shuffle(list_IDs)

        self.list_IDs = list_IDs
        self.leaf_ins = settings.get('leaf_ins', False)
        self.node_reor = settings.get('node_reor', False)
        self.target_var = target_var

        IDs = [ID + self.variant + self.structureless + self.test for ID in self.list_IDs]

        self.cascades = []
        for ID in IDs:
            self.cascades.append(ld(ID, self.data_dir + 'graphs/'))
        self.x_size = self.cascades[0].X.shape[1]

        self.cascades_proc = []
        for c in self.cascades:
            self.cascades_proc.append(c.retrieve_data(self.target_var, self.leaf_ins, self.node_reor, self.emo_data))

        del self.cascades
        

    def __len__(self):
        # Denotes the total number of samples
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Generates one sample of data
        
        ID = self.list_IDs[index]

        g, v, emo = self.cascades_proc[index]

        return th.Tensor([int(ID)]), g, v, emo

    def sample(self, ids):
        # sample ids

        # load cascade "embedding"
        df = pd.read_csv(self.data_dir + 'grouped' + self.variant + '.csv')
        # select embeddings of cascades to load with the dataloader
        df = df[df.cascade_id.isin(ids)].reset_index()
        # drop emotions
        X = df.iloc[:, 2:12].values
        y = df.veracity.values

        sampler = IterativeSampler()

        # retireve only the indices in the original data frame
        print('sampling')
        # retireve indices AS POSITION IN THE DF, NOT CASCADE IDS of cascades
        # to sample, ignore sampled data X and labels y
        _, _, d = sampler.fit_sample(X, y)

        # get casde ids from indices in the df
        ids = df.loc[list(d.values()), 'cascade_id'].astype(str).tolist()

        return ids
        # alternative : return ids without doubles
        # return list(set(ids)
 
    def get_emo_data(self, data_file):
        """
        Function that retrieves the mean and covariance of the root features
        of the true and false cascades. Used for root feature perturbation 
        """
        
        cols = ['sadness', 'anticipation', 'disgust', 'surprise', 'anger', 'joy', 'fear', 'trust']
        
        df = pd.read_csv('../data/grouped.csv')
        
        X_pos = df[df.veracity == 1][cols].values
        X_neg = df[df.veracity == 0][cols].values
        
        dpos = {'mean': np.mean(X_pos, axis=0), 'cov': np.cov(X_pos, rowvar=0)}
        dneg = {'mean': np.mean(X_neg, axis=0), 'cov': np.cov(X_neg, rowvar=0)}
        
        return {1: dpos, 0: dneg}
