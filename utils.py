import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import config
import json
from matplotlib import ticker

def save_parameters(opts, save_name):
    cur_dict = opts.__dict__
    cur_dict = {k: v for k, v in cur_dict.items() if k[:2] != '__'}
    with open(save_name + '.json', 'w') as f:
        json.dump(cur_dict, f)

def load_parameters(save_path):
    """Load config."""
    with open(os.path.join(save_path + '.json'), 'r') as f:
        config_dict = json.load(f)

    class Config():
        pass

    config = Config()
    for key, val in config_dict.items():
        setattr(config, key, val)
    return config

def sort_weights(mat, axis, arg_pos= 1):
    if arg_pos:
        temp = np.square(mat * (mat > 0))
    else:
        temp = np.square(mat * (mat < 0))

    if axis == 1:
        ix = np.arange(mat.shape[1]).reshape(-1, 1)
        moment = np.matmul(temp, ix).squeeze() / np.sum(temp, axis=1)
        sort_ix = np.argsort(moment)
        mat_sorted = mat[sort_ix, :]
    else:
        ix = np.arange(mat.shape[0]).reshape(1, -1)
        moment = np.matmul(ix, temp).squeeze() / np.sum(temp, axis=0)
        sort_ix = np.argsort(moment)

        max_ix = np.argmax(mat, axis=0)
        sort_ix = np.argsort(max_ix)
        mat_sorted = mat[:, sort_ix]
    return sort_ix, mat_sorted

def pretty_image(tup, col, row, save_name):
    """input: list of tuples
    first arg of each tuple: title
    second arg of each tuple: matrix to plot
    """
    c, r = 0, 0
    fig, ax = plt.subplots(nrows=row, ncols=col)
    for t, w in tup:
        if np.ndim(w) == 1:
            w = w.reshape(1,-1)
        sns.heatmap(w, cmap='RdBu_r', vmin=-1, vmax=1, ax=ax[r, c], cbar=False)
        if t != '':
            ax[r, c].set_title(t)
        ax[r, c].axis('off')
        ax[r, c].axis('image')
        c += 1
        if c >= col:
            r += 1
            c = 0
    fig.savefig(save_name, bbox_inches='tight', figsize = (14,10))
    plt.close()

def pretty_plot(tup, col, row, save_name):
    """input: list of tuples
    first arg of each tuple: title
    second arg of each tuple: matrix to plot
    """
    c, r = 0, 0
    fig, ax = plt.subplots(nrows=row, ncols=col)
    for t, w in tup:
        ax[r, c].plot(w)
        ax[r, c].set_title(t)
        c += 1
        if c >= col:
            r += 1
            c = 0
    fig.savefig(save_name, bbox_inches='tight', figsize= (14,10))
    plt.close()

def adjust(ax):
    plt.style.use('dark_background')
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    # ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    # ax.yaxis.set_ticks([0, 0.5, 1.0])
    # ax.set_ylim([0, 1])


