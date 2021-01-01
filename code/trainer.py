import copy
import torch as th

from callbacks import CallbackContainer, History, Printer
from utils import *
import math

from metrics import MetricCallback, MetricContainer
from model import DeepTreeLSTM

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
        self._has_metrics = False
        


    def compile(self, optimizer_tree, optimizer_top, criterion, scheduler_tree=None, scheduler_top=None, callbacks=None ,metrics=None):

        self._criterion = criterion
        self._optimizer_tree = optimizer_tree
        self._lr_tree = optimizer_tree
        self._optimizer_top = optimizer_top

        if scheduler_tree is not None:
            self.scheduler_tree = scheduler_tree
            self._has_Scheduler_tree = True

        if scheduler_top is not None:
            self.scheduler_top = scheduler_top
            self._has_Scheduler_top = True

        self.metrics = metrics
        if metrics is not None:
            self._has_metrics = True
            self.metric_container = MetricContainer(metrics)
            self.val_metric_container = MetricContainer(metrics, 'val_')

        self.history = History()
        self.callbacks = [self.history]
        if callbacks is not None:
            self.callbacks = [self.history] + callbacks
            


    def fit(self, loader, val_loader=None, num_epoch=50, cuda_device=-1, verbose=1):

        best_loss = 1e15
        self.best_model_state = None

        self.model.train(mode=True)

        device = set_device(cuda_device)

        self._stop_training = False

        len_inputs = len(loader.sampler) if loader.sampler else len(loader.dataset)
        batch_size = loader.batch_size
        num_batches = int(math.ceil(len_inputs / batch_size))

        tmp_callbacks = []
        
        if verbose > 0:
            tmp_callbacks.append(Printer(self.metrics))

        if self._has_metrics:
            tmp_callbacks.append(MetricCallback(self.metric_container))
            tmp_callbacks.append(MetricCallback(self.val_metric_container))

        callback_container = CallbackContainer(self.callbacks + tmp_callbacks)
        callback_container.set_trainer(self)

        train_begin_log = loader.dataset.log
        train_begin_log.update(self.model.log)
        train_begin_log.update({'num_epoch': num_epoch,
                                'batch_size': batch_size,
                                'num_batches': num_batches})
        callback_container.on_train_begin(train_begin_log)

        epoch_logs = {}
        for epoch in range(num_epoch):
            callback_container.on_epoch_begin(epoch)
            
            batch_logs = {}
            batch_logs['loss'] = 0.
            samples_seen = 0

            for step, batch in enumerate(loader):
                callback_container.on_batch_begin(step)

                self._optimizer_top.zero_grad()
                self._optimizer_tree.zero_grad()
                y, y_hat = self.__learn(batch, device)
                loss = self._criterion(y_hat, y)
                loss.backward()
                self._optimizer_top.step()
                self._optimizer_tree.step()

                batch_size = batch.X.shape[0]
                batch_logs['loss'] = (batch_logs['loss']*samples_seen + loss.item()*batch_size) / (samples_seen + batch_size)
                samples_seen += batch_size

                if self._has_metrics:
                    metrics_log = self.metric_container(y_hat, y)
                    batch_logs.update(metrics_log)

                callback_container.on_batch_end(step, batch_logs)

            if val_loader is not None:
                eval_logs = self.evaluate(val_loader, cuda_device, verbose=0, best=False)
                epoch_logs.update(eval_logs)
                if eval_logs['val_loss'] < best_loss:
                    best_loss = eval_logs['val_loss']
                    self.best_model_state = copy.deepcopy(self.model.state_dict())

            if self._has_Scheduler_tree:
                self.scheduler_tree.step()

            if self._has_Scheduler_top:
                self.scheduler_top.step()
            
            epoch_logs.update(batch_logs)

            callback_container.on_epoch_end(epoch, epoch_logs)

            if self._stop_training:
                break

        self.model.train(mode=False)
        callback_container.on_train_end(epoch_logs)


    def evaluate(self, loader, cuda_device=-1, verbose=1, best=True):
        self.model.train(mode=False)

        if best and self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        device = set_device(cuda_device)

        eval_logs = {'val_loss': 0.}
        samples_seen = 0

        for batch in loader:
            self._optimizer_tree.zero_grad()
            self._optimizer_top.zero_grad()
            y, y_hat = self.__learn(batch, device)
            loss = self._criterion(y_hat, y)
            
            batch_size = batch.X.shape[0]
            eval_logs['val_loss'] = (eval_logs['val_loss']*samples_seen + loss.item()*batch_size) / (samples_seen + batch_size)
            samples_seen += batch_size

            if self._has_metrics:
                metric_logs = self.val_metric_container(y_hat, y)
                eval_logs.update(metric_logs)
            


        self.model.train(mode=True)

        return eval_logs
        

    def predict(self, loader, cuda_device=-1, verbose=1, best=True):
        self.model.train(mode=False)

        if best and self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        device = set_device(cuda_device)
        out_list = []

        for batch in loader:
            g = batch.graph.to(device)
            n = g.number_of_nodes()
            h = th.zeros((n, self.h_size)).to(device)
            c = th.zeros((n, self.h_size)).to(device)
            out = self.model.predict(batch, g, h, c)
            out_list.append(out.detach())

        final_preds = th.cat(out_list, dim=0)
        self.model.train(mode=True)
        return final_preds


    def __learn(self, batch, device):
        g = batch.graph.to(device)
        n = g.number_of_nodes()
        h = th.zeros((n, self.h_size)).to(device)
        c = th.zeros((n, self.h_size)).to(device)
        y_hat = self.model(batch, g, h, c)
        return batch.y.to(device), y_hat