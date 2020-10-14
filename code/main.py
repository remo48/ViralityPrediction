import os, sys

from callbacks import EarlyStopping, ExperimentLogger

try : os.chdir(os.path.dirname(sys.argv[0]))
except : pass

sys.path.append('../code/')

import numpy as np
import torch as th
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from datetime import datetime
from utils import *
from model import DeepTreeLSTMClassifier, DeepTreeLSTMRegressor
from dataset import CascadeData
from trainer import DeepTreeTrainer

default_settings = {
    'h_size': 24,
    'top_sizes': (32,64,32)
}

def main(args):

    verb = args.verbose
    data_dir = '../data/'
    graphs_dir = '../data/graphs/'
    out_dir = '../results/'

    cuda_id = args.cuda_id
    if cuda_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
        if verb==2: print("There are %d devices available" % th.cuda.device_count())
        
    device = set_device(cuda_id)

    settings = {
        'h_size': args.h_size,
        'top_sizes': (32, 64, 32),
        'p_drop': args.p_drop,
        'lr_tree': args.lr_tree,
        'decay_tree': args.decay_tree,
        'lr_top': args.lr_top,
        'decay_top': args.decay_top,
        'bi': bool(args.bi),
        'deep': bool(args.deep),
        'lead_ins': args.leaf_ins,
        'node_reor': args.node_reor,
        'variant': args.variant,
        'structureless': args.structureless
    }

    train_ids = np.array([ID.split('.')[0] for ID in os.listdir(graphs_dir) if '_' not in ID])
    test_ids = np.unique([ID.split('_')[0] for ID in os.listdir(graphs_dir) if 'test' in ID])

    train_set = CascadeData(train_ids, 'cascade_size_log', data_dir, settings)
    test_set = CascadeData(test_ids, 'cascade_size_log', data_dir, settings, test = True)

    train_generator = DataLoader(train_set, collate_fn=cascade_batcher(device), batch_size= args.batch_size, num_workers=8)
    test_generator = DataLoader(test_set, collate_fn=cascade_batcher(device), batch_size= args.batch_size, num_workers=8)


    x_size = args.x_size
    if args.structureless: 
        x_size = x_size - 2
    emo_size = 8
    epochs = args.epochs

    deep_tree = DeepTreeLSTMRegressor(x_size, emo_size, settings)

    criterion = nn.MSELoss().to(device)
    optimizer_tree = optim.SGD(deep_tree.bottom_net.parameters(), lr = settings['lr_tree'], weight_decay = settings['decay_tree'])
    optimizer_top = optim.SGD(deep_tree.top_net.parameters(), lr = settings['lr_top'], weight_decay = settings['decay_top'])
    scheduler_tree = optim.lr_scheduler.StepLR(optimizer_tree, step_size=10, gamma=0.8)
    scheduler_top = optim.lr_scheduler.StepLR(optimizer_top, step_size=10, gamma=0.8)


    callbacks = [EarlyStopping(patience=1),
                 ExperimentLogger(out_dir, settings=settings)]

    model_trainer = DeepTreeTrainer(deep_tree)
    model_trainer.compile(optimizer_tree, optimizer_top, criterion, scheduler_tree=scheduler_tree, scheduler_top=scheduler_top, callbacks=callbacks, metrics=['mae'])
    model_trainer.fit(train_generator, test_generator, epochs, cuda_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--x_size', dest = 'x_size', default = 12, type = int)
    parser.add_argument('--h_size', dest = 'h_size', default = 24, type = int)
    parser.add_argument('--sample', dest = 'sample', action='store_true')
    parser.add_argument('--bsize', dest = 'batch_size', default = 25, type = int)
    parser.add_argument('--pd', dest = 'p_drop', default = 0.1, type = float)
    parser.add_argument('--epochs', dest = 'epochs', default = 50, type = int)
    parser.add_argument('--lr_tree', dest = 'lr_tree', default = 0.01, type = float)
    parser.add_argument('--lr_top', dest = 'lr_top', default = 0.01, type = float)
    parser.add_argument('--decay_tree', dest = 'decay_tree', default = 0.003, type = float)
    parser.add_argument('--decay_top', dest = 'decay_top', default = 0.006, type = float)
    parser.add_argument('--cuda', dest='cuda_id', default=-1)
    parser.add_argument('--leaf_ins', action='store_true', default = False)
    parser.add_argument('--node_reor', action='store_true', default = False)
    parser.add_argument('--emo_pert', action='store_true', default = False)
    parser.add_argument('--verbose', type=int, default = 1)
    parser.add_argument('--save_ids', action='store_true', default = False)
    parser.add_argument('--deep', default = 1, type=int)
    parser.add_argument('--bi', default = 1, type=int)
    parser.add_argument('--variant', type = str, default = '')
    parser.add_argument('--structureless', action='store_true', default = False)
    parser.add_argument('--log', action='store_true', default = False)

    args = parser.parse_args()

    main(args)