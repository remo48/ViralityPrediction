import torch as th
import torch.nn as nn
import dgl
from utils import *
import math


class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)
        self.d = nn.Dropout(0.2)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_tild = th.sum(nodes.mailbox['h'], 1)
        f = th.sigmoid(self.U_f(nodes.mailbox['h']))
        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_tild), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        return {'h': h, 'c': c}

class TreeLSTM(nn.Module):
    def __init__(self,
                 x_size,
                 h_size):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.cell = ChildSumTreeLSTMCell(x_size, h_size)

    def forward(self, batch, g, h, c):
        """Compute tree-lstm prediction given a batch.

        Parameters
        ----------
        batch : dgl.data.SSTBatch
            The data batch.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.

        Returns
        -------
        out
        """
        g.ndata['iou'] = self.cell.W_iou(batch.X)
        g.ndata['h'] = h
        g.ndata['c'] = c
        # propagate
        dgl.prop_nodes_topo(g, 
                            message_func=self.cell.message_func,
                            reduce_func=self.cell.reduce_func,
                            apply_node_func=self.cell.apply_node_func
                            )

        h = g.ndata.pop('h')
        
        # indexes of root nodes
        head_ids = th.nonzero(batch.isroot, as_tuple=False).flatten()
        # h of root nodes
        head_h = th.index_select(h, 0, head_ids)
        lims_ids = head_ids.tolist() + [g.number_of_nodes()]
        # average of h of non root node by tree
        inner_h = th.cat([th.mean(h[s+1:e,:],dim=0).view(1,-1) for s, e in zip(lims_ids[:-1],lims_ids[1:])])
        out = th.cat([head_h, inner_h], dim = 1)
        #out = head_h
        return out

class BiDiTreeLSTM(nn.Module):
    def __init__(self,
                 x_size,
                 h_size):
        super(BiDiTreeLSTM, self).__init__()
        self.x_size = x_size
        self.cell_bottom_up = ChildSumTreeLSTMCell(x_size, h_size)
        self.cell_top_down = ChildSumTreeLSTMCell(x_size+h_size, h_size)

    def propagate(self, g, cell, X, h, c):   
        g.ndata['iou'] = cell.W_iou(X)
        g.ndata['h'] = h
        g.ndata['c'] = c
        # propagate
        dgl.prop_nodes_topo(g,
                            message_func=cell.message_func,
                            reduce_func=cell.reduce_func,
                            apply_node_func=cell.apply_node_func)

        return g.ndata.pop('h')

    def forward(self, batch, g, h, c):
        """Compute tree-lstm prediction given a batch.

        Parameters
        ----------
        batch : dgl.data.SSTBatch
            The data batch.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.

        Returns
        -------
        out
        """
        h_bottom_up = self.propagate(g, self.cell_bottom_up, batch.X, h, c)
        
        g_rev = dgl.reverse(g)
        #g_rev = dgl.batch([gu.reverse() for gu in dgl.unbatch(batch.graph)])
        h_top_down = self.propagate(g_rev, self.cell_top_down, th.cat([batch.X, h_bottom_up], dim = 1), h, c)
        
        
        # indexes of root nodes
        root_ids = th.nonzero(batch.isroot, as_tuple=False).flatten()
        # h of root nodes
        root_h_bottom_up = th.index_select(h_bottom_up, 0, root_ids)
        
        # limit of ids of trees in graphs batch
        lims_ids = root_ids.tolist() + [g.number_of_nodes()]
        
        trees_h = [h_top_down[s:e,:] for s, e in zip(lims_ids[:-1],lims_ids[1:])]
        trees_isleaf = [batch.isleaf[s:e] for s, e in zip(lims_ids[:-1],lims_ids[1:])]
        leaves_h_top_down = th.cat([th.mean(th.index_select(tree, 0, th.nonzero(leaves, as_tuple=False).flatten()),dim=0).view(1,-1) for (tree, leaves) in zip(trees_h, trees_isleaf)], dim = 0)
                
        out = th.cat([root_h_bottom_up, leaves_h_top_down], dim = 1)
        #out = leaves_h_top_down
        return out

class DeepTreeLSTM(nn.Module):
    '''
    Base Class for CascadeLSTM Regressor & Classifier
    '''
    def __init__(self, x_size, h_size, bi, deep):
        self.h_size = h_size
        
        super(DeepTreeLSTM, self).__init__()
        
        if bi :
            net = BiDiTreeLSTM(x_size, h_size)
        else :
            net = TreeLSTM(x_size, h_size)
            
            
        self.deep = deep    
        self.bottom_net = net
        self.top_net = None

        
    def forward(self, batch, g, h, c):
        h_top = self.bottom_net.forward(batch, g, h, c)
        if self.deep:
            X2 = th.cat([h_top, batch.emo], dim = 1)
        else:
            X2 = h_top
        out = self.top_net(X2)
        return out


class DeepTreeLSTMRegressor(DeepTreeLSTM):
    def __init__(self, x_size, h_size, emo_size, top_sizes, pd, bi, deep):
        super().__init__(x_size, h_size, bi, deep)

        net = nn.Sequential()
        
        if self.deep:
        
            net.add_module('d_in', nn.Dropout(p=pd))
            net.add_module('l_in', nn.Linear(2*h_size+emo_size, top_sizes[0]))
            net.add_module('tf_in', nn.ReLU())

            for i, (in_dim, out_dim) in enumerate(zip(top_sizes[:-1], top_sizes[1:])):
                net.add_module('d' + str(i), nn.Dropout(p=pd))
                net.add_module('l' + str(i), nn.Linear(in_dim, out_dim))
                net.add_module('tf' + str(i), nn.ReLU())

            net.add_module('d_out', nn.Dropout(p=pd))
            net.add_module('l_out', nn.Linear(top_sizes[-1],1))

        else:
            net.add_module('d', nn.Dropout(p=pd))
            net.add_module('l', nn.Linear(2*h_size, 1))

        self.top_net = net

class DeepTreeLSTMClassifier(DeepTreeLSTM):
    def __init__(self, x_size, h_size, num_classes, emo_size, top_sizes, pd, bi, deep, logit=True):
        super().__init__(x_size, h_size, bi, deep)

        net = nn.Sequential()
        
        if self.deep:
        
            net.add_module('d_in', nn.Dropout(p=pd))
            net.add_module('l_in', nn.Linear(2*h_size+emo_size, top_sizes[0]))
            net.add_module('tf_in', nn.ReLU())

            for i, (in_dim, out_dim) in enumerate(zip(top_sizes[:-1], top_sizes[1:])):
                net.add_module('d' + str(i), nn.Dropout(p=pd))
                net.add_module('l' + str(i), nn.Linear(in_dim, out_dim))
                net.add_module('tf' + str(i), nn.ReLU())

            net.add_module('d_out', nn.Dropout(p=pd))
            net.add_module('l_out', nn.Linear(top_sizes[-1], num_classes))
            if logit:
                net.add_module('tf_out', nn.Sigmoid())

        else:
            net.add_module('d', nn.Dropout(p=pd))
            net.add_module('l', nn.Linear(2*h_size, num_classes))

        self.top_net = net


class DeepTreeTrainer:
    '''
    class for high level training of DeepTreeLSTM models
    '''
    def __init__(self, model):
        if not isinstance(model, DeepTreeLSTM):
            raise ValueError('model argument must inherit from DeepTreeLSTM')
        
        self.model = model
        init_net(self.model)
        
        self.h_size = model.h_size
        self._has_Scheduler_tree = False
        self._has_Scheduler_top = False
        


    def compile(self, optimizer_tree, optimizer_top, criterion, scheduler_tree=None, scheduler_top=None, metrics=None):

        self._criterion = criterion
        self._optimizer_tree = optimizer_tree
        self._optimizer_top = optimizer_top

        if scheduler_tree is not None:
            self.scheduler_tree = scheduler_tree
            self._has_Scheduler_tree = True

        if scheduler_top is not None:
            self.scheduler_top = scheduler_top
            self._has_Scheduler_top = True


    def fit(self, loader, val_loader=None, num_epoch=50, cuda_device=-1, verbose=1):

        self.model.train(mode=True)

        device = set_device(cuda_device)

        len_inputs = len(loader.sampler) if loader.sampler else len(loader.dataset)
        batch_size = loader.batch_size

        num_batches = int(math.ceil(len_inputs / batch_size))


        for epoch in range(num_epoch):
            pbar = Pbar(num_batches, epoch, num_epoch)

            
            for step, batch in enumerate(loader):

                self._optimizer_top.zero_grad()
                self._optimizer_tree.zero_grad()
                y, y_hat = self.__learn(batch, device)
                loss = self._criterion(y, y_hat)
                loss.backward()
                self._optimizer_top.step()
                self._optimizer_tree.step()

                pbar.update(step, [('loss', loss.item())])

            if self._has_Scheduler_tree:
                self.scheduler_tree.step()

            if self._has_Scheduler_top:
                self.scheduler_top.step()

            if val_loader is not None:
                loss = self.evaluate(val_loader, cuda_device, verbose)

                pbar.add_val([('val_loss', loss.item())])

        self.model.train(mode=False)


    def evaluate(self, loader, cuda_device=-1, verbose=1):
        self.model.train(mode=False)

        loss = None
        device = set_device(cuda_device)

        for batch in loader:
            self._optimizer_tree.zero_grad()
            self._optimizer_top.zero_grad()
            y, y_hat = self.__learn(batch, device)
            loss = self._criterion(y, y_hat)

        self.model.train(mode=True)

        return loss
        

    def predict(self):
        pass


    def __learn(self, batch, device):
        g = batch.graph.to(device)
        n = g.number_of_nodes()
        h = th.zeros((n, self.h_size)).to(device)
        c = th.zeros((n, self.h_size)).to(device)
        y_hat = self.model(batch, g, h, c)
        return batch.y.to(device), y_hat