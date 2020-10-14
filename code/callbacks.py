import datetime
import time
import sys
import os

def _get_current_time():
    return datetime.datetime.now().strftime("%B %d, %Y - %I:%M%p")

class CallbackContainer():
    """
    Container holding a list of callbacks.
    """
    def __init__(self, callbacks=None, queue_length=10):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_trainer(self, trainer):
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        logs['start_time'] = _get_current_time()
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or {}
        #logs['final_loss'] = self.trainer.history.epoch_losses[-1],
        #logs['best_loss'] = min(self.trainer.history.epoch_losses),
        logs['stop_time'] = _get_current_time()
        for callback in self.callbacks:
            callback.on_train_end(logs)

class callback:
    '''
    abstract class for callback functions
    '''

    def __init__(self):
        pass

    def set_params(self, params):
        self.params = params

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

class History(callback):
    """
    Callback that records loss and metrics history into a `History` object.
    """
    def __init__(self, model):
        self.trainer = model

    def on_train_begin(self, logs=None):
        self.epoch_metrics = {}

    def on_epoch_end(self, epoch, logs=None):
        for k in logs:
            if k not in self.epoch_metrics:
                self.epoch_metrics[k] = [logs[k]]
            else:
                self.epoch_metrics[k].append(logs[k])

    def __getitem__(self, name):
        return self.epoch_metrics[name]

    def __repr__(self):
        return str(self.epoch_metrics)

    def __str__(self):
        return str(self.epoch_metrics)

class Printer(callback):

    def __init__(self, metrics, width=40, verbose=1):
        self.width = width
        self.verbose = verbose
        self.metrics = metrics

    def on_train_begin(self, logs):
        self.num_epoch = logs['num_epoch']
        self.num_batches = logs['num_batches']

    def on_train_end(self, logs):
        pass

    def on_epoch_begin(self, epoch, logs):
        self.start = time.time()
        title = f'Epoch: {epoch+1}/{self.num_epoch}\n'

        sys.stdout.write(title)
        sys.stdout.flush()

    def on_epoch_end(self, epoch, logs):
        sys.stdout.write('\r')

        bar = f'{self.num_batches}/{self.num_batches} '
        bar += '[' + '#'*self.width + '] '

        sys.stdout.write(bar)

        elapsed = time.time() - self.start
        time_per_step = elapsed / self.num_batches

        if time_per_step >= 1 or time_per_step == 0:
                time_per_step_format = ' %.0fs/step' % (time_per_step)
        elif time_per_step >= 1e-3:
            time_per_step_format = ' %.0fms/step' % (time_per_step * 1e3)
        else:
            time_per_step_format = ' %.0fus/step' % (time_per_step * 1e6)

        if elapsed > 3600:
            time_format = '%d:%02d:%02d' % (elapsed // 3600,
                                           (elapsed % 3600) // 60, elapsed % 60)
        elif elapsed > 60:
            time_format = '%d:%02d' % (elapsed // 60, elapsed % 60)
        else:
            time_format = '%ds' % elapsed

        time_format += time_per_step_format
        
        info = '- ' + time_format + ' '

        info += f'- loss: {logs["loss"]:.4f} '

        for metric in self.metrics:
            s = 'val_' + metric
            info += f'- {s}: {logs[s]:.4f} '

        info += '\n'
        sys.stdout.write(info)
        sys.stdout.flush()
        

    def on_batch_end(self, batch, logs):
        now = time.time()
        left = (batch+1) * self.width // self.num_batches
        right = self.width - left

        sys.stdout.write('\r')

        bar = f'{batch+1}/{self.num_batches} '
        bar += '[' + '#'*left + ' '*right + '] '
        sys.stdout.write(bar)

        time_per_step = (now - self.start) / (batch+1)

        eta = (self.num_batches - batch) * time_per_step
        if eta > 3600:
            eta_format = '%d:%02d:%02d' % (eta // 3600,
                                        (eta % 3600) // 60, eta % 60)
        elif eta > 60:
            eta_format = '%d:%02d' % (eta // 60, eta % 60)
        else:
            eta_format = '%ds' % eta

        info  = f'- ETA: {eta_format} '

        info += f'- loss: {logs["loss"]:.4f} '

        sys.stdout.write(info)
        sys.stdout.flush()

class EarlyStopping(callback):
    '''
        Callback class for early stopping

        TODO: not working yet
    '''
    def __init__(self, monitor='val_loss', min_delta=0, patience=10):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
    
    def on_train_begin(self, logs):
        self.wait = 0
        self.best = 1e15

    def on_epoch_end(self, epoch, logs):
        current = logs[self.monitor]

        if (self.best - current) > self.min_delta:
            self.wait = 1
            self.best = current
        
        else:
            if self.wait > self.patience:
                self.trainer._stop_training = True
            self.wait += 1

class ExperimentLogger(callback):

    def __init__(self, directory, filename='logs_final.csv', settings=None):
        self.directory = directory
        self.filename = filename
        self.file = os.path.join(self.directory, self.filename)

        if settings is not None:
            self.settings = settings
