import numpy as np
from torch import nn as nn
import torch
import os
import json
import matplotlib.pyplot as plt
import config

class Abstract_Model(nn.Module):
    """Abstract Model class."""

    def __init__(self, save_path):
        super(Abstract_Model, self).__init__()

        if save_path is None:
            save_path = os.getcwd()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = save_path
        self.config = None


    def save(self, name = 'model', epoch=None):
        save_path = self.save_path
        if epoch is not None:
            save_path = os.path.join(save_path, 'epoch', str(epoch).zfill(4))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model_path = os.path.join(save_path, name + '.pth')
        torch.save(self.state_dict(), model_path)
        self.save_config(save_path)

        print("Model saved in path: %s" % model_path)

    def load(self, name = 'model', epoch=None):
        save_path = self.save_path
        if epoch is not None:
            save_path = os.path.join(save_path, 'epoch', str(epoch).zfill(4))
        save_path = os.path.join(save_path, name + '.pth')
        self.load_state_dict(torch.load(save_path))

        print("Model restored from path: {:s}".format(save_path))

    def save_config(self, save_path):
        config_dict = self.config.__dict__
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config_dict, f)

        with open(os.path.join(save_path, 'config.txt'), "w") as f:
            for k, v in config_dict.items():
                f.write(str(k) + ' >>> ' + str(v) + '\n\n')


class Simple_Model(Abstract_Model):
    def __init__(self, config, isize, osize):
        super(Simple_Model, self).__init__(config.save_path)

        self.hidden_size = config.rnn_size
        self.batch_size = config.batch_size
        self.config = config

        self.i2h = nn.Linear(isize, config.rnn_size)
        self.h_w = torch.nn.Parameter(.01 * torch.rand(config.rnn_size, config.rnn_size))
        self.h_b = torch.nn.Parameter(.01 * torch.rand(config.rnn_size))
        mask = np.ones((config.rnn_size, config.rnn_size)).astype(np.float32)
        np.fill_diagonal(mask, 0)
        mask = torch.from_numpy(mask)
        self.h_mask = torch.nn.Parameter(mask)
        self.h2o = torch.nn.Linear(config.rnn_size, osize)


    def forward(self, input, hidden):
        i = self.i2h(input)
        effective_h_w = torch.mul(self.h_w, self.h_mask)
        h = torch.matmul(hidden, effective_h_w)
        hidden = torch.relu(i + h)
        out = self.h2o(hidden)
        return hidden, out

    def initialZeroState(self):
        return torch.zeros(self.batch_size, self.hidden_size)


def load_config(save_path, epoch=None):
    if epoch is not None:
        save_path = os.path.join(save_path, 'epoch', str(epoch).zfill(4))

    with open(os.path.join(save_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)

    c = config.shared_config()
    for key, val in config_dict.items():
        setattr(c, key, val)
    return c