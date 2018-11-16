import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import config
from utils import adjust
import utils
import os
import inputs
import pickle as pkl
from analysis import sort_weights

def plot_weights(data, save_path, ylabel ='', xlabel= ''):
    rect = [0.15, 0.15, 0.65, 0.65]
    rect_cb = [0.82, 0.15, 0.02, 0.65]
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(4.0, 4.0))
    ax = fig.add_axes(rect)

    utils.adjust(ax)
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

def plot_weights_simple(data, save_path, ylabel='',xlabel='', vmin=-1.,
                        vmax=1.):
    rect = [0.15, 0.15, 0.65, 0.65]
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(4.0, 4.0))
    ax = fig.add_axes(rect)

    utils.adjust(ax)
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
    plt.savefig(os.path.join('./lab_meeting/images/' + save_path + '.png'),
                transparent=True)

def plot_stationary_weights(opts):
    save_path = opts.save_path
    weight_name = opts.weight_name

    with open(os.path.join(save_path, weight_name + '.pkl'), 'rb') as f:
        weight_dict = pkl.load(f)

    W_h = weight_dict['model/hidden/W_h:0']
    plot_weights(W_h, 'W_h_initial')

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
        W_h_ab_sorted = W_h_ab[:, sort_ix]
        W_h_ba_sorted = W_h_ba[sort_ix, :]
    else:
        W_h_ab_bb = weight_dict['model/hidden/W_h_ab_bb:0']
        W_h_ab = W_h_ab_bb[:state_size,:]
        W_h_ba = weight_dict['model/hidden/W_h_ba:0']
        W_i_b = weight_dict['model/input/W_i_b:0']
        W_h_aa = weight_dict['model/hidden/W_h_aa:0']
        W_i_b_sorted = W_i_b[:, sort_ix]

        W_h_ab_sorted = W_h_ab[:, sort_ix]
        W_h_ba_sorted = W_h_ba[sort_ix, :]

        left = np.vstack((W_h_aa, W_h_ba_sorted))
        right = np.vstack((W_h_ab_sorted, W_h_ab_bb[state_size:,:]))
        W_h = np.hstack((left,right))

    #hack
    if W_i_b_sorted.shape[0]>2:
        W_i_b_sorted = W_i_b_sorted[2:4,:]

    plot_weights_simple(W_h, 'W_h')
    plot_weights_simple(W_h_aa, 'W_h_aa')
    plot_weights_simple(W_h_ab_sorted, 'W_h_ab_sorted',vmin=-.5, vmax=.5)
    plot_weights_simple(W_h_ba_sorted, 'W_h_ba_sorted',vmin=-.5, vmax=.5)
    if not stationary:
        plot_weights_simple(W_i_b_sorted, 'W_i_b_sorted',vmin=-5, vmax=5)

def plot_input_output_weights_stationary():
    input = np.eye(20)
    hzeros = np.zeros((20, 15))
    vzeros = np.zeros((15,20))
    hidden = np.random.uniform(-3,3, [35, 35])
    W_i_b = np.random.uniform(-3, 3, [2, 40])
    plot_weights_simple(hidden,'W_h_random')
    plot_weights_simple(np.hstack((input, hzeros)),'W_input')
    plot_weights_simple(np.vstack((input, vzeros)),'W_output')
    plot_weights_simple(W_i_b, 'W_i_b_initial')
    plot_weights_simple(np.array([[1, 0]]), 'activity_velocity')
    plot_weights_simple(np.random.uniform(-3, 3, [30, 1]),
                        'activity_support')




if __name__ == '__main__':
    mpl.rcParams['font.size'] = 12

    plot_input_output_weights_stationary()

    opts = utils.load_parameters('./gold_copy/stationary/parameters')
    opts.save_path = './gold_copy/stationary/'
    plot_stationary_weights(opts)

    opts = utils.load_parameters('./gold/non_stationary/parameters')
    opts.save_path = './gold/non_stationary/'
    sort_ix = sort_weights(opts)
    plot_nonstationary_weights(opts, sort_ix)