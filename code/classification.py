import os, sys
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch as th

sys.path.append('../code/')
from dataset import CascadeData
from utils import get_class_weights, cascade_batcher
from model import DeepTreeLSTMClassifier
from callbacks import EarlyStopping, ExperimentLogger
from trainer import DeepTreeTrainer

try : os.chdir(os.path.dirname(sys.argv[0]))
except : pass

def main():
    data_dir = '../data/'
    out_dir = '../results/'
    graphs_dir = data_dir + 'graphs_all/'
    cascade_size_file = data_dir + 'cascade_size.csv'
    
    device = th.device('cpu')
    variant = '1_hour'
    batch_size = 25
    x_size = 12
    h_size = 8
    emo_size = 8
    top_size = (16,16)
    p_drop = 0.1
    lr_tree = 0.005
    lr_top = 0.001
    decay_tree = 0.003
    decay_top = 0.006

    train_ids = np.array([ID.split('_')[0] for ID in os.listdir(graphs_dir) if variant in ID and 'test' not in ID])
    test_ids = np.unique([ID.split('_')[0] for ID in os.listdir(graphs_dir) if variant + '_test' in ID])

    train_set = CascadeData(train_ids, graphs_dir, cascade_size_file, variant=variant, categorical=True)
    test_set = CascadeData(test_ids, graphs_dir, cascade_size_file, test=True, variant=variant, categorical=True)

    weights, weights_all = get_class_weights(train_set)
    weighted_sampler = WeightedRandomSampler(weights_all, len(weights_all))

    train_generator = DataLoader(train_set, collate_fn=cascade_batcher(device), batch_size=batch_size, num_workers=8, sampler=weighted_sampler)
    test_generator = DataLoader(test_set, collate_fn=cascade_batcher(device), batch_size=batch_size, num_workers=8)

    deep_tree = DeepTreeLSTMClassifier(x_size, 4, emo_size, h_size=h_size, top_sizes=top_size, pd=p_drop)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer_tree = th.optim.Adam(deep_tree.bottom_net.parameters(), lr = lr_tree, weight_decay = decay_tree)
    optimizer_top = th.optim.Adam(deep_tree.top_net.parameters(), lr = lr_top, weight_decay = decay_top)
    scheduler_tree = th.optim.lr_scheduler.StepLR(optimizer_tree, step_size=10, gamma=0.8)
    scheduler_top = th.optim.lr_scheduler.StepLR(optimizer_top, step_size=10, gamma=0.8)


    callbacks = [EarlyStopping(patience=10),
                    ExperimentLogger(out_dir, filename='logs_test.csv')]

    model_trainer = DeepTreeTrainer(deep_tree)
    model_trainer.compile(optimizer_tree, optimizer_top, criterion, scheduler_tree=scheduler_tree, scheduler_top=scheduler_top, callbacks=callbacks, metrics=['mul_acc'])
    model_trainer.fit(train_generator, test_generator, 10)

if __name__ == "__main__":
    main()