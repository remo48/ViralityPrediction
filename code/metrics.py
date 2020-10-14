from numpy.core.fromnumeric import mean
from sklearn.metrics import mean_absolute_error
from callbacks import callback

class MetricContainer:
    
    def __init__(self, metrics, prefix=''):
        self.metrics = []
        self.prefix = prefix

        for metric in metrics:
            if metric == 'mae':
                self.metrics.append(MeanAbsoluteError())

    def __call__(self, output_batch, target_batch):
        metric_logs = {}

        for metric in self.metrics:
            metric_logs[self.prefix + metric.name] = metric(output_batch.detach(), target_batch.detach())

        return metric_logs

class MetricCallback(callback):
    def __init__(self, metric_container):
        self.metric_container = metric_container
        self.metrics = self.metric_container.metrics

    def on_epoch_begin(self, epoch, logs):
        for metric in self.metrics:
            metric.reset()

class Metric(object):

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