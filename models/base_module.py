import pdb

import torch

class BaseModule(torch.nn.Module):

    def __init__(self):
        super(BaseModule, self).__init__()

    def forward(self):
        raise NotImplementedError

    def parameters(self):
        pass

    def step(self):
        for key, value in self.optimizers.items():
            if 'freeze' not in self.net_conf[key] or self.net_conf[key]['freeze'] == False:
                value.step()

    def train(self):
        for key, value in self.net.items():
            value.train()

    def eval(self):
        for key, value in self.net.items():
            value.eval()

    def to(self, device):
        for key, value in self.net.items():
            self.net[key] = value.to(device)
        return self

    def zero_grad(self):
        for key, value in self.net.items():
            value.zero_grad()

    def get_net_state(self):
        net_state = {}
        for key, value in self.net.items():
            net_state[key] = value.state_dict()

        return net_state

    def load_net_state(self, state):
        for key, value in self.net.items():
            if key in state:
                value.load_state_dict(state[key], strict=False)

