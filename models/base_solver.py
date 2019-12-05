import os
import pdb
import importlib
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import tqdm

def _print_conf(conf):
    print('\n-----------------------------------------\n')
    print('solver configuration:')
    for key, value in conf['solver_conf'].items():
        print(key + ': ' + str(value))

    print('\n-----------------------------------------\n')
    print('net configuration:')
    for key, value in conf['net_conf'].items():
        print(str(key) + ': ' + str(value))

    print('\n-----------------------------------------\n')
    print('loader configuration:')
    for key, value in conf['loader_conf'].items():
        print(key + ': ' + str(value))
    print('\n-----------------------------------------\n')

def get_solver_from_solver_state(solver_state_path):
    solver_state = torch.load(open(solver_state_path, 'rb'))
    conf = solver_state['conf']
    solver_path = conf['solver_conf']['solver_path']
    solver_name = solver_path.split('.')[-1]
    solver = importlib.import_module(solver_path).get_solver(conf)
    solver.load_solver_state(solver_state)

    print('Sucessfully load solver state from {}!'.format(solver_state_path))
    _print_conf(conf)

    return solver

class BaseSolver:
    """
    A solver can do network training, evaluation and test
    """
    def __init__(self, conf):
        self.conf = conf
        self.solver_conf = conf['solver_conf']
        self.net_conf = conf['net_conf']
        self.loader_conf = conf['loader_conf']
        self.checkpoints_dir = self.solver_conf['checkpoints_dir']

        self.max_epoch = self.solver_conf['max_epoch']
        self.cur_epoch = 1
        self.global_step = 0
        self.init_network()
        self.init_optimizers()
        self.init_best_checkpoint_settings()
        self.init_tensors()
        self.summary_writer = SummaryWriter(log_dir=self.solver_conf['log_dir'])
        self.load_to_gpu()
        _print_conf(conf)


    def init_network(self):
        """
        Initialize the network model
        """
        net_path = self.net_conf['net_path']
        self.nets = importlib.import_module(net_path).get_model(self.net_conf)

    def init_optimizers(self):
        optimizers = {}
        for key, value in self.nets.net.items():
            parameters = filter(lambda x: x.requires_grad, value.parameters())

            try:
                next(parameters)
            except StopIteration:
                continue
            finally:
                parameters = filter(lambda x: x.requires_grad, value.parameters())
            lr = self.net_conf['lr_conf']['init_lr']

            if self.net_conf['optimizer_name'] == 'SGD':
                optimizers[key] = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
            elif self.net_conf['optimizer_name'] == 'Adam':
                optimizers[key] = torch.optim.Adam(parameters, lr=lr, betas=(0.5, 0.999))
            else:
                raise ValueError
        self.optimizers = optimizers

    def get_optimizers(self):
        return self.optimizers


    def train(self, train_loader, val_loader=None):
        """
        Train and test the network model with the given train and test data loader
        """
        start_epoch = self.cur_epoch
        for self.cur_epoch in range(start_epoch, self.max_epoch + 1):
            self.adjust_lr(self.cur_epoch)

            train_state = self.process_epoch(train_loader, 'train')
            self.update_checkpoint(self.cur_epoch)

            if val_loader is not None:
                eval_state = self.process_epoch(val_loader, 'val')
                self.update_best_checkpoint(eval_state, self.cur_epoch)


    def test(self, val_loader):
        # self.load_to_gpu()
        eval_state = self.process_epoch(val_loader, 'val')
        return eval_state

    def get_solver_state(self):
        state = {}

        nets_state = self.nets.get_net_state()
        optimizers_state = {}
        for key, value in self.optimizers.items():
            optimizers_state[key] = value.state_dict()

        state['cur_epoch'] = self.cur_epoch
        state['conf'] = self.conf
        state['global_step'] = self.global_step
        state['nets_state'] = nets_state
        state['optimizers_state'] = optimizers_state

        return state

    def load_solver_state(self, state):
        if self.solver_conf['load_epoch']:
            self.cur_epoch = state['cur_epoch'] + 1
            self.global_step = state['global_step']

        self.nets.load_net_state(state['nets_state'])

        if 'optimizers_state' in state:
            for key, value in self.optimizers.items():
                self.optimizers[key].load_state_dict(state['optimizers_state'][key])

    def update_checkpoint(self, cur_epoch):
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        path = os.path.join(self.checkpoints_dir, 'network_' + str(cur_epoch) + '.pkl')
        solver_state = self.get_solver_state()
        torch.save(solver_state, open(path, 'wb'))

        old_path_network = os.path.join(self.checkpoints_dir, 'network_' + str(cur_epoch-1) + '.pkl')
        if os.path.isfile(old_path_network) and (cur_epoch) % 10 != 0:
            os.remove(old_path_network)

    def init_best_checkpoint_settings(self):
        self.metric_names = self.solver_conf['metric_names']
        self.metric_values = {}
        self.best_epoch = None

    def update_best_checkpoint(self, eval_state, cur_epoch):
        for metric_name in self.metric_names:
            if 'scalar|'+metric_name not in eval_state.keys():
                return
            cur_metric_value = eval_state['scalar|'+metric_name]
            if metric_name not in self.metric_values.keys() or cur_metric_value > self.metric_values[metric_name]:
                self.metric_values[metric_name] = cur_metric_value
                self.best_epoch = cur_epoch

                path = os.path.join(self.checkpoints_dir, 'network_best_' + metric_name + '.pkl')
                cur_solver_state = self.get_solver_state()
                torch.save(cur_solver_state, open(path, 'wb'))

    def process_epoch(self, data_loader, phase='train'):
        tq = tqdm.tqdm(data_loader)
        tq.set_description('{} | Epoch: {} | Step: {}'.format(phase, self.cur_epoch, self.global_step))

        epoch_state = {}
        for idx, batch in enumerate(tq):
            cur_state = self.process_batch(batch, phase)
            self.global_step = self.global_step + 1

            if idx % 20 == 0:
                self.summary_write_state(cur_state, phase)
                for key, value in cur_state.items():
                    if key.split('|')[0] == 'scalar':
                        if key not in epoch_state:
                            epoch_state[key] = 0
                        epoch_state[key] = epoch_state[key] + cur_state[key]
            tq.set_postfix({'psnr': cur_state['scalar|psnr']})

        tq.close()

        for key, value in epoch_state.items():
            epoch_state[key] = epoch_state[key] / len(data_loader)

        return epoch_state

    def summary_write_state(self, state, phase='train'):
        for key, value in state.items():
            prefix, name = key.split('|')
            if prefix == 'scalar':
                self.summary_writer.add_scalar(phase+'_'+name, value, global_step=self.global_step)
            elif prefix == 'image':
                self.summary_writer.add_image(phase+'_'+name, value, global_step=self.global_step)
            elif prefix == 'graph':
                self.summary_writer.add_graph(*value)

    def adjust_lr(self, cur_epoch):

        lr_conf = self.net_conf['lr_conf']
        init_lr = lr_conf['init_lr']
        if lr_conf['decay_type'] == 'linear':
            end_lr = self.net_conf['end_lr']
            start_decay_epoch = self.net_conf['start_decay_epoch']
            end_decay_epoch = self.net_conf['end_decay_epoch']
            decay_per_epoch = (end_lr - init_lr) / (end_decay_epoch - start_decay_epoch)
            cur_lr = init_lr - max(0, cur_epoch - start_decay_epoch) * decay_per_epoch

        elif lr_conf['decay_type'] == 'expo':
            decay_base = lr_conf['decay_base']
            cur_lr = init_lr * (decay_base ** (self.cur_epoch - 1))

        else:
            raise ValueError

        self.cur_lr = cur_lr
        for name, optimizer in self.optimizers.items():
            for param_group in optimizer.param_groups:
                param_group['lr'] = cur_lr

    def process_batch(self, batch, phase='train'):
        raise NotImplementedError

    def init_tensors(self):
        self.tensors = {}
        raise NotImplementedError

    def set_tensors(self, batch):
        raise NotImplementedError

    def load_to_gpu(self):
        gpu_ids = self.solver_conf['gpu_ids']
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

        for key, value in self.nets.net.items():
            self.nets.net[key] = nn.DataParallel(value.cuda())
        # self.nets = nn.DataParallel(self.nets.cuda())

        # for key, value in self.optimizers.items():
        #     self.optimizers[key] = nn.DataParallel(value.cuda())

        for key, tensor in self.tensors.items():
            self.tensors[key] = self.tensors[key].cuda()
