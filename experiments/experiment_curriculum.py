"""A collection of experiments."""

import os
import train
from collections import OrderedDict
import numpy as np
import config


def vary_weight_loss(i):
    root_dir = '/Users/peterwang/Desktop/PYTHON/Neural_head_direction/experiments/vary_weight_loss/'
    stationary_opts = config.stationary_model_config()
    stationary_opts.save_path = os.path.join(root_dir, 'files', str(i).zfill(2), 'stationary')
    stationary_opts.rnn_size = 60
    stationary_opts.load_checkpoint = False
    stationary_opts.epoch = 500
    stationary_opts.time_steps = 25
    stationary_opts.time_loss_start = 5
    stationary_opts.time_loss_end = 25
    stationary_opts.activity_alpha = .1

    moving_opts = config.non_stationary_model_config()
    moving_opts.save_path = os.path.join(root_dir, 'files', str(i).zfill(2), 'moving')
    moving_opts.load_checkpoint = False
    moving_opts.dir_weights = os.path.join(stationary_opts.save_path, 'weight.pkl')
    moving_opts.initialize_W_ab_diagonal = False
    moving_opts.initialize_W_ba_diagonal = False
    moving_opts.epoch = 500
    moving_opts.velocity_use = 2
    moving_opts.velocity_max = 2
    moving_opts.weight_alpha = 1
    moving_opts.activity_alpha = .1

    hp_ranges = OrderedDict()
    hp_ranges['weight_alpha'] = [0, .05, .1, .2, .4, .8, 1]
    return stationary_opts, moving_opts, hp_ranges


def varying_config(experiment, i):
    # Ranges of hyperparameters to loop over
    stationary_opts, moving_opts, hp_ranges = experiment(i)

    # Unravel the input index
    keys = hp_ranges.keys()
    dims = [len(hp_ranges[k]) for k in keys]
    n_max = np.prod(dims)
    indices = np.unravel_index(i % n_max, dims=dims)

    if i >= n_max:
        return

    # Set up new hyperparameter
    for key, index in zip(keys, indices):
        setattr(stationary_opts, key, hp_ranges[key][index])
    train.train(stationary_opts)
    train.train(moving_opts)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for i in range(0,100):
        varying_config(vary_weight_loss, i)