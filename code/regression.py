import os, sys 
import argparse
import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append('../code/')

from dataset import CascadeData
from utils import cascade_batcher, set_device
from model import DeepTreeLSTMRegressor
from callbacks import EarlyStopping, ModelLogger, ExperimentLogger
from trainer import DeepTreeTrainer

try : os.chdir(os.path.dirname(sys.argv[0]))
except : pass


def regression(variant = '1_hour',
                structureless = False,
                batch_size = 8,
                x_size = 12,
                h_size = 8,
                emo_size = 8,
                top_sizes = (16,16),
                p_drop = 0.1,
                verbose = 1,
                bi = True,
                deep = True,
                lr_tree = 0.05,
                lr_top = 0.01,
                decay_tree = 0.003,
                decay_top = 0.006,
                epochs = 60,
                cuda_id = -1):


    data_dir = '../data/'
    out_dir = '../results/'
    graphs_dir = data_dir + 'graphs/'
    cascade_size_file = data_dir + 'cascade_size.csv'

    device = set_device(cuda_id)

    if structureless:
        x_size = x_size - 2

    train_ids = np.array([ID.split('_')[0] for ID in os.listdir(graphs_dir) if variant in ID and 'test' not in ID])
    test_ids = np.unique([ID.split('_')[0] for ID in os.listdir(graphs_dir) if variant + '_test' in ID])

    train_set = CascadeData(train_ids, graphs_dir, cascade_size_file, variant=variant, structureless=structureless)
    test_set = CascadeData(test_ids, graphs_dir, cascade_size_file, variant=variant, structureless=structureless, test=True)

    train_generator = DataLoader(train_set, collate_fn=cascade_batcher(device), batch_size=batch_size, num_workers=8)
    test_generator = DataLoader(test_set, collate_fn=cascade_batcher(device), batch_size=batch_size, num_workers=8)

    deep_tree = DeepTreeLSTMRegressor(x_size, emo_size, h_size=h_size, top_sizes=top_sizes, bi=bi, deep=deep, pd=p_drop)

    criterion = nn.MSELoss().to(device)
    optimizer_tree = th.optim.SGD(deep_tree.bottom_net.parameters(), lr = lr_tree, weight_decay = decay_tree)
    optimizer_top = th.optim.SGD(deep_tree.top_net.parameters(), lr = lr_top, weight_decay = decay_top)
    scheduler_tree = th.optim.lr_scheduler.StepLR(optimizer_tree, step_size=10, gamma=0.8)
    scheduler_top = th.optim.lr_scheduler.StepLR(optimizer_top, step_size=10, gamma=0.8)

    callbacks = [EarlyStopping(patience=10), ModelLogger(out_dir+'models/'), ExperimentLogger(out_dir, 'logs_regression.csv')]

    model_trainer = DeepTreeTrainer(deep_tree)
    model_trainer.compile(optimizer_tree, optimizer_top, criterion, scheduler_tree=scheduler_tree, scheduler_top=scheduler_top, callbacks=callbacks, metrics=['mae'])
    model_trainer.fit(train_generator, test_generator, epochs, cuda_id, verbose)
    return deep_tree

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage='%(prog)s [options]', description='Single run of Cascade-LSTM Regression')

    parser.add_argument('--x_size', default = 12, type = int, help='the size of the input tensor')
    parser.add_argument('--h_size', default = 8, type = int, help='the size of a hidden layer cell')
    parser.add_argument('--bsize', dest = 'batch_size', default = 8, type = int, help='batch size used for training')
    parser.add_argument('--pd', dest = 'p_drop', default = 0.1, type = float, help='dropout value of top layer')
    parser.add_argument('--epochs', default = 60, type = int, help='number of epochs')
    parser.add_argument('--lr_tree', default = 0.05, type = float, help='learning rate for tree component')
    parser.add_argument('--lr_top', default = 0.01, type = float, help='learning rate for top layer component')
    parser.add_argument('--decay_tree', default = 0.003, type = float, help='weight decay for tree component')
    parser.add_argument('--decay_top', default = 0.006, type = float, help='weight decay for tree component')
    parser.add_argument('--cuda', dest='cuda_id', default=-1, help='id of cuda device used for training (-1 for cpu)')
    parser.add_argument('--verbose', type=int, default = 1, help='verbosity: 0: no output, 1: full output')
    parser.add_argument('--rootless', dest='deep', action='store_false', help='use root-less model variation')
    parser.add_argument('--uni', dest='bi', action='store_false', help='use unidirectional model variation')
    parser.add_argument('--variant', type = str, default = '1_hour', help='lifetime of cascades used for training')
    parser.add_argument('--structureless', action='store_true', help='use structureless model variation')

    args = parser.parse_args()

    regression(variant = args.variant,
                structureless = args.structureless,
                batch_size = args.batch_size,
                x_size = args.x_size,
                h_size = args.h_size,
                p_drop = args.p_drop,
                bi = args.bi,
                deep = args.deep,
                lr_tree = args.lr_tree,
                lr_top = args.lr_top,
                decay_tree = args.decay_tree,
                decay_top = args.decay_top,
                epochs = args.epochs,
                cuda_id = args.cuda_id,
                verbose = args.verbose)