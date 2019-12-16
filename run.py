import importlib
import argparse

import torch

from datasets.aligned_dataset import AlignedDataset
from datasets.data_loader import MyDataLoader
from models import base_solver

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, default=None)
    parser.add_argument('--state_path', type=str, default=None)

    args = parser.parse_args()

    if args.state_path is not None:
        solver = base_solver.get_solver_from_solver_state(args.state_path)
        solver_conf = solver.conf['solver_conf']
        loader_conf = solver.conf['loader_conf']

    else:
        ## Load parameters ##
        conf = importlib.import_module(args.conf_path).conf
        solver_conf = conf['solver_conf']
        loader_conf = conf['loader_conf']
        solver_path = solver_conf['solver_path']
        solver = importlib.import_module(solver_path).get_solver(conf)
        if solver_conf['load_state']:
            solver.load_solver_state(torch.load(solver_conf['solver_state_path']))


    train_dataset = AlignedDataset(loader_conf, phase='train')
    val_dataset = AlignedDataset(loader_conf, phase='val')

    train_loader = MyDataLoader(train_dataset, loader_conf, phase='train').get_loader()
    val_loader = MyDataLoader(val_dataset, loader_conf, phase='val').get_loader()

    if solver_conf['phase'] == 'train':
        solver.train(train_loader, val_loader)
    else:
        state = solver.test(val_loader)
        imgs = state['image|image']

