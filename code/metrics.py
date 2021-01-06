from numpy.core.fromnumeric import mean
from sklearn.metrics import mean_absolute_error, roc_auc_score
from callbacks import callback
import numpy as np
import torch as th

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class MetricContainer:
    '''
    Container for all metrics used during training.

    Args:
        metrics: list of string names of metrics to use
        prefix: prefix for metric names (e.g. val_)
    '''
    
    def __init__(self, metrics, prefix=''):
        self.metrics = []
        self.prefix = prefix

        for metric in metrics:
            if metric == 'mae':
                self.metrics.append(MeanAbsoluteError())
            elif metric == 'acc':
                self.metrics.append(Accuracy())
            elif metric == 'auc':
                self.metrics.append(AUC())
            elif metric == 'precision':
                self.metrics.append(Precision())
            elif metric == 'mul_acc':
                self.metrics.append(Multi_Accuracy())

    def __call__(self, output_batch, target_batch):
        metric_logs = {}

        for metric in self.metrics:
            metric_logs[self.prefix + metric.name] = metric(output_batch.detach(), target_batch.detach())

        return metric_logs

class MetricCallback(callback):
    '''
    Callback function, which ensures that metrics are calculated during training
    '''

    def __init__(self, metric_container):
        self.metric_container = metric_container
        self.metrics = self.metric_container.metrics

    def on_epoch_begin(self, epoch, logs):
        for metric in self.metrics:
            metric.reset()

class Metric(object):
    '''
    Abstract class for Metrics
    '''

    def __call__(self, y_pred, y_true):
        raise NotImplementedError('Custom Metrics must implement this function')

    def reset(self):
        raise NotImplementedError('Custom Metrics must implement this function')

class MeanAbsoluteError(Metric):
    def __init__(self):
        self.name = 'mae'
        self.seen_sample = 0
        self.mae = 0

    def reset(self):
        self.seen_sample = 0
        self.mae = 0

    def __call__(self, y_pred, y_true):
        size = len(y_pred)
        self.mae = (self.mae * self.seen_sample + mean_absolute_error(y_pred, y_true) * size) / (self.seen_sample + size)
        self.seen_sample += size
        return self.mae

class AUC(Metric):
    def __init__(self):
        self.name = 'auc'
        self.y_pred = th.Tensor()
        self.y_true = th.Tensor()

    def reset(self):
        self.y_pred = th.Tensor()
        self.y_true = th.Tensor()

    def __call__(self, y_pred, y_true):
        y_pred = sigmoid(y_pred)
        self.y_pred = th.cat((self.y_pred, y_pred))
        self.y_true = th.cat((self.y_true, y_true))
        return roc_auc_score(self.y_true, self.y_pred)

class Accuracy(Metric):
    def __init__(self):
        self.name = 'acc'
        self.seen_sample = 0
        self.correct_sample = 0

    def reset(self):
        self.seen_sample = 0
        self.correct_sample = 0

    def __call__(self, y_pred, y_true):
        y_pred = sigmoid(y_pred)
        y_pred_round = y_pred.round().long()
        self.correct_sample += y_pred_round.eq(y_true).float().sum()
        self.seen_sample += len(y_pred)
        accuracy = float(self.correct_sample) / float(self.seen_sample)
        return accuracy

class Multi_Accuracy(Metric):
    def __init__(self):
        self.name = 'mul_acc'
        self.seen_sample = 0
        self.correct_sample = 0

    def reset(self):
        self.seen_sample = 0
        self.correct_sample = 0

    def __call__(self, y_pred, y_true):
        y_pred_softmax = th.softmax(y_pred, dim = 1)
        y_pred_tags = y_pred_softmax.argmax(dim=1)
        self.correct_sample += (y_pred_tags == y_true).float().sum()
        self.seen_sample += len(y_pred)
        accuracy = float(self.correct_sample) / float(self.seen_sample)
        return accuracy

class Precision(Metric):
    def __init__(self):
        self.name = 'precision'
        self.predicted_positive = 0
        self.true_positive = 0

    def reset(self):
        self.predicted_positive = 0
        self.true_positive = 0

    def __call__(self, y_pred, y_true):
        y_pred_round = sigmoid(y_pred).round().float()
        self.predicted_positive += y_pred_round.sum()
        self.true_positive += y_pred_round.logical_and(y_true).sum()
        precision = float(self.true_positive) / float(self.predicted_positive)
        return precision
