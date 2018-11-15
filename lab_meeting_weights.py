import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import seaborn as sns
import config
from utils import adjust
import utils
import os
import inputs
import pickle as pkl
from analysis import sort_weights



def plot_weights(data, title, save_path):
    rect = [0.15, 0.15, 0.65, 0.65]
    rect_cb = [0.82, 0.15, 0.02, 0.65]
    fig = plt.figure(figsize=(4.0, 4.0))
    ax = fig.add_axes(rect)
    vlim = np.round(np.max(abs(data)), decimals=1)
    im = ax.imshow(data, cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                   interpolation='none')
    plt.axis('tight')
    plt.title(title)
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_xlabel('To PNs', labelpad=-5)
    ax.set_ylabel('From ORNs', labelpad=-5)
    ax.set_xticks([0, data.shape[0]])
    ax.set_yticks([0, data.shape[1]])
    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(im, cax=ax, ticks=[-vlim, vlim])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Weight', labelpad=-10)
    plt.tick_params(axis='both', which='major')
    plt.axis('tight')
    # utils.adjust(ax)
    plt.savefig(os.path.join('./lab_meeting/' + save_path + '.png'),
                transparent=True)

def plot_stationary_weights(opts):
    save_path = opts.save_path
    weight_name = opts.weight_name

    with open(os.path.join(save_path, weight_name + '.pkl'), 'rb') as f:
        weight_dict = pkl.load(f)

    W_h = weight_dict['model/hidden/W_h:0']
    plot_weights(W_h, 'weights','weights')

def plot_nonstationary_weights(opts, sort_ix):
    """Visualization of trained network."""
    stationary = opts.stationary
    state_size = opts.state_size
    save_path = opts.save_path
    weight_name = opts.weight_name

    with open(os.path.join(save_path, weight_name + '.pkl'), 'rb') as f:
        weight_dict = pkl.load(f)

    # plot sorted_weights
    if stationary:
        W_h = weight_dict['model/hidden/W_h:0']
        W_h_ab = W_h[:state_size, state_size:]
        W_h_ba = W_h[state_size:, :state_size]
        W_i_b_sorted = None
        W_h_aa = W_h[:state_size, :state_size]
    else:
        W_h_ab_bb = weight_dict['model/hidden/W_h_ab_bb:0']
        W_h_ab = W_h_ab_bb[:state_size,:]
        W_h_ba = weight_dict['model/hidden/W_h_ba:0']
        W_i_b = weight_dict['model/input/W_i_b:0']
        W_h_aa = weight_dict['model/hidden/W_h_aa:0']
        W_i_b_sorted = W_i_b[:, sort_ix]

    W_h_ab_sorted = W_h_ab[:, sort_ix]
    W_h_ba_sorted = W_h_ba[sort_ix, :]

    plot_weights(W_h_aa, 'aa', 'W_h_aa')
    plot_weights(W_h_ab, 'Unsorted', 'W_h_ab_unsorted')
    plot_weights(W_h_ab_sorted, 'Sorted', 'W_h_ab_sorted')
    plot_weights(W_h_ba, 'Unsorted', 'W_h_ba_unsorted')
    plot_weights(W_h_ba_sorted, 'sorted', 'W_h_ba_sorted')
    if not stationary:
        plot_weights(W_i_b, 'Unsorted', 'W_i_b_unsorted')
        plot_weights(W_i_b_sorted, 'Unsorted', 'W_i_b_unsorted')

if __name__ == '__main__':
    mpl.rcParams['font.size'] = 7
    opts = utils.load_parameters('./lab_meeting/080/non_stationary/parameters')
    sort_ix = sort_weights(opts)
    plot_nonstationary_weights(opts, sort_ix)