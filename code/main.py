import os, sys

from callbacks import EarlyStopping, ExperimentLogger, ModelLogger

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
from model import DeepTreeLSTMRegressor
from dataset import CascadeData
from trainer import DeepTreeTrainer

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

    train_ids = np.array([ID.split('_')[0] for ID in os.listdir(graphs_dir) if args.variant in ID and 'test' not in ID])
    test_ids = np.unique([ID.split('_')[0] for ID in os.listdir(graphs_dir) if args.variant + '_test' in ID])

    train_set = CascadeData(train_ids, data_dir, variant=args.variant)
    test_set = CascadeData(test_ids, data_dir, variant=args.variant, test=True)

    train_generator = DataLoader(train_set, collate_fn=cascade_batcher(device), batch_size=args.batch_size, num_workers=8)
    test_generator = DataLoader(test_set, collate_fn=cascade_batcher(device), batch_size=args.batch_size, num_workers=8)

    # r_train = class_ratio(train_generator)

    x_size = args.x_size
    if args.structureless: 
        x_size = x_size - 2
    emo_size = 8
    epochs = args.epochs

    deep_tree = DeepTreeLSTMRegressor(x_size, emo_size, h_size=args.h_size, top_sizes=(16,16), bi=args.bi)

    criterion = nn.MSELoss().to(device)
    optimizer_tree = optim.SGD(deep_tree.bottom_net.parameters(), lr = args.lr_tree, weight_decay = args.decay_tree)
    optimizer_top = optim.SGD(deep_tree.top_net.parameters(), lr = args.lr_top, weight_decay = args.decay_top)
    scheduler_tree = optim.lr_scheduler.StepLR(optimizer_tree, step_size=10, gamma=0.8)
    scheduler_top = optim.lr_scheduler.StepLR(optimizer_top, step_size=10, gamma=0.8)


    callbacks = [EarlyStopping(patience=10), ModelLogger(out_dir+'models/'), ExperimentLogger(out_dir, 'logs_new.csv')]

    model_trainer = DeepTreeTrainer(deep_tree)
    model_trainer.compile(optimizer_tree, optimizer_top, criterion, scheduler_tree=scheduler_tree, scheduler_top=scheduler_top, callbacks=callbacks, metrics=['mae'])
    model_trainer.fit(train_generator, test_generator, epochs, cuda_id)
    
    ys = []
    for batch in test_generator:
        ys.append(batch.y)

    y = th.cat(ys, dim=0).flatten().tolist()
    y_pred = model_trainer.predict(test_generator).flatten().tolist()
    df_res = pd.DataFrame({'id': test_ids, 'y_true' : y, 'y_pred' : y_pred})
    df_res.to_csv('../results/preds.csv', index=False, header=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--x_size', dest = 'x_size', default = 12, type = int)
    parser.add_argument('--h_size', dest = 'h_size', default = 10, type = int)
    parser.add_argument('--sample', dest = 'sample', action='store_true')
    parser.add_argument('--bsize', dest = 'batch_size', default = 10, type = int)
    parser.add_argument('--pd', dest = 'p_drop', default = 0.15, type = float)
    parser.add_argument('--epochs', dest = 'epochs', default = 10, type = int)
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
    parser.add_argument('--variant', type = str, default = '1_hour')
    parser.add_argument('--structureless', action='store_true', default = False)
    parser.add_argument('--log', action='store_true', default = False)

    args = parser.parse_args()

    main(args)