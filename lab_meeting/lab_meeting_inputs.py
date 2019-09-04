import numpy as np
import matplotlib.pyplot as plt
import config
from utils import adjust
import os
from datasets import inputs


def plot_moving_inputs(inputs, labels, opts):
    rc = (2,3)
    fig, ax = plt.subplots(rc[0], rc[1])
    state = [x[:,:opts.state_size] for x in inputs[:rc[0]]]
    extra = [x[:,opts.state_size:opts.state_size+2] for x in inputs[:rc[0]]]
    labels = labels[:rc[0]]

    i=0
    for batch in zip(state, extra, labels):
        for d in batch:
            plot_ix = np.unravel_index(i, rc)
            cur_ax = ax[plot_ix]
            adjust(cur_ax)
            plt.sca(cur_ax)
            plt.imshow(d, cmap='RdBu_r', vmin=-.3, vmax=.3)
            if i%3 !=1:
                plt.xticks([0, 19])
            else:
                plt.xticks([])
            plt.yticks([0, 49])
            cb = plt.colorbar()
            cb.set_ticks([0, .3])
            i+=1
    path = os.path.join('./lab_meeting/images', 'input_non_stationary')
    plt.savefig(path + '.png', dpi=300)

def plot_stationary_inputs(inputs, labels, opts):
    rc = (2,2)
    fig, ax = plt.subplots(rc[0], rc[1])
    state = inputs[:rc[0]]
    labels = labels[:rc[0]]
    i=0
    for batch in zip(state, labels):
        for d in batch:
            plot_ix = np.unravel_index(i, rc)
            cur_ax = ax[plot_ix]
            adjust(cur_ax)
            plt.sca(cur_ax)
            # cbarBoo = True if i %2==1 else True
            plt.imshow(d, cmap='RdBu_r', vmin=-.3, vmax=.3)
            plt.xticks([0, 19])
            plt.yticks([0, 49])
            cb = plt.colorbar()
            cb.set_ticks([0, .3])
            i+=1
    path = os.path.join('./lab_meeting/images', 'input_stationary')
    plt.savefig(path + '.png', dpi=300)



if __name__ == '__main__':
    stationary = config.stationary_input_config()
    non_stationary = config.non_stationary_input_config()
    stationary.time_steps = 50
    non_stationary.time_steps = 50
    x, y = inputs.create_inputs(stationary)
    plot_stationary_inputs(x, y, stationary)
    x, y= inputs.create_inputs(non_stationary)
    plot_moving_inputs(x,y, non_stationary)
