import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
import tensorflow as tf

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

def get_tf_vars_as_dict():
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    var_dict = {os.path.split(v.name)[1][:-2]: v for v in vars}
    return var_dict

def sort_weights(mat, axis, arg_pos= 1):
    if arg_pos:
        temp = np.square(mat * (mat > 0))
    else:
        temp = np.square(mat * (mat < 0))

    if axis == 1:
        max_ix = np.argmax(mat, axis=1)
        sort_ix = np.argsort(max_ix)
        mat_sorted = mat[sort_ix, :]
    else:
        max_ix = np.argmax(mat, axis=0)
        sort_ix = np.argsort(max_ix)
        mat_sorted = mat[:, sort_ix]
    return sort_ix, mat_sorted

def pretty_image(tup, col, row, save_name, vmin = -1, vmax = 1):
    """input: list of tuples
    first arg of each tuple: title
    second arg of each tuple: matrix to plot
    """
    c, r = 0, 0
    fig, ax = plt.subplots(nrows=row, ncols=col)
    for t, w in tup:
        if np.ndim(w) == 1:
            w = w.reshape(1,-1)
        sns.heatmap(w, cmap='RdBu_r', vmin=vmin, vmax=vmax, ax=ax[r, c],
                    cbar=False)
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

def plot_weights(data, save_path, ylabel ='', xlabel= ''):
    rect = [0.15, 0.15, 0.65, 0.65]
    rect_cb = [0.82, 0.15, 0.02, 0.65]
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(4.0, 4.0))
    ax = fig.add_axes(rect)

    adjust(ax)
    vlim = np.round(np.max(abs(data)), decimals=1)
    im = ax.imshow(data, cmap= 'RdBu_r', vmin=-vlim, vmax=vlim,
                   interpolation='none')
    plt.axis('tight')
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_xlabel(xlabel, labelpad=-5)
    ax.set_ylabel(ylabel, labelpad=-5)
    ax.set_xticks([0, data.shape[1]])
    ax.set_yticks([0, data.shape[0]])
    plt.axis('scaled')
    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(im, cax=ax, ticks=[-vlim, vlim])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Weight', labelpad=-10)
    plt.tick_params(axis='both', which='major')
    plt.savefig(os.path.join('./lab_meeting/images/' + save_path + '.png'),
                transparent=True)

def plot_weights_simple(data, save_path, ylabel='',xlabel='', vmin= None, vmax= None):
    rect = [0.15, 0.15, 0.65, 0.65]
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(4.0, 4.0))
    ax = fig.add_axes(rect)

    adjust(ax)
    if vmin is None or vmax is None:
        vmax = np.round(np.max(abs(data)), decimals=1)
        vmin = -vmax

    im = ax.imshow(data, cmap= 'RdBu_r', vmin=vmin, vmax=vmax,
                   interpolation='none')
    plt.axis('tight')
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_xlabel(xlabel, labelpad=-5)
    ax.set_ylabel(ylabel, labelpad=-5)
    ax.set_xticks([0, data.shape[1]-1])
    ax.set_yticks([0, data.shape[0]-1])
    plt.tick_params(axis='both', which='major')
    # plt.axis('image')
    plt.savefig(save_path + '.png',
                transparent=True)


