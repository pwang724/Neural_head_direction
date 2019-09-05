"""A collection of experiments."""

import os
from old import train
from collections import OrderedDict
import numpy as np
import config


def vary_batch(i):
    opts = config.stationary_model_config()
    opts.save_path = '/Users/peterwang/Desktop/PYTHON/Neural_head_direction/experiments' \
                     '/vary_batch/files/' + str(i).zfill(2)
    opts.rnn_size = 60
    opts.load_checkpoint = False
    opts.epoch = 300
    opts.time_steps = 25
    opts.time_loss_start = 5
    opts.time_loss_end = 25

    hp_ranges = OrderedDict()
    hp_ranges['batch_size'] = [5,10,20,25,40,50,100]
    return opts, hp_ranges

def vary_time_loss_start(i):
    opts = config.stationary_model_config()
    opts.save_path = '/Users/peterwang/Desktop/PYTHON/Neural_head_direction/experiments' \
                     '/vary_time_loss_start/files/' + str(i).zfill(2)
    opts.rnn_size = 60
    opts.load_checkpoint = False
    opts.epoch = 300
    opts.time_steps = 25
    opts.batch_size = 20
    opts.time_loss_start = 5
    opts.time_loss_end = 25

    hp_ranges = OrderedDict()
    hp_ranges['time_loss_start'] = [0,2,4,6,8,10,15]
    return opts, hp_ranges



def varying_config(experiment, i):
    # Ranges of hyperparameters to loop over
    opts, hp_ranges = experiment(i)

    # Unravel the input index
    keys = hp_ranges.keys()
    dims = [len(hp_ranges[k]) for k in keys]
    n_max = np.prod(dims)
    indices = np.unravel_index(i % n_max, dims=dims)

    if i >= n_max:
        return

    # Set up new hyperparameter
    for key, index in zip(keys, indices):
        setattr(opts, key, hp_ranges[key][index])
    train.train(opts)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for i in range(0,100):
        varying_config(vary_time_loss_start, i)