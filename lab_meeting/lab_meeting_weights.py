import numpy as np
import matplotlib as mpl
import utils
from utils import plot_weights_simple
import os
import pickle as pkl
from analysis.basics import sort_weights

path = '../lab_meeting/images/'

def plot_stationary_weights(opts, sort_ix):
    save_path = opts.save_path
    weight_name = opts.weight_name

    with open(os.path.join(save_path, weight_name + '.pkl'), 'rb') as f:
        weight_dict = pkl.load(f)
    W_h_aa = weight_dict['model/hidden/W_h_aa:0']
    W_h_bb = weight_dict['model/hidden/W_h_bb:0']
    W_h_ab = weight_dict['model/hidden/W_h_ab:0']
    W_h_ba = weight_dict['model/hidden/W_h_ba:0']
    W_h_ab_sorted = W_h_ab[:, sort_ix]
    W_h_ba_sorted = W_h_ba[sort_ix, :]
    left = np.vstack((W_h_aa, W_h_ba_sorted))
    right = np.vstack((W_h_ab_sorted, W_h_bb))
    W_h = np.hstack((left, right))
    plot_weights_simple(W_h, path + 'W_h_initial')

def plot_nonstationary_weights(opts, sort_ix):
    """Visualization of trained network."""
    stationary = opts.stationary
    state_size = opts.state_size
    save_path = opts.save_path
    weight_name = opts.weight_name

    with open(os.path.join(save_path, weight_name + '.pkl'), 'rb') as f:
        weight_dict = pkl.load(f)

    # plot sorted_weights
    W_h_aa = weight_dict['model/hidden/W_h_aa:0']
    W_h_bb = weight_dict['model/hidden/W_h_bb:0']
    W_h_ab = weight_dict['model/hidden/W_h_ab:0']
    W_h_ba = weight_dict['model/hidden/W_h_ba:0']
    W_h_ab_sorted = W_h_ab[:, sort_ix]
    W_h_ba_sorted = W_h_ba[sort_ix, :]

    left = np.vstack((W_h_aa, W_h_ba_sorted))
    right = np.vstack((W_h_ab_sorted, W_h_bb))
    W_h_sorted = np.hstack((left,right))

    left = np.vstack((W_h_aa, W_h_ba))
    right = np.vstack((W_h_ab, W_h_bb))
    W_h = np.hstack((left,right))

    if opts.stationary == 0:
        W_i_b = weight_dict['model/input/W_i_b:0']
        W_i_b_sorted = W_i_b[:, sort_ix]

    vmin = -1
    vmax = 1
    plot_weights_simple(W_h_sorted, path + 'W_h_sorted',vmin = vmin, vmax=vmax)
    plot_weights_simple(W_h, path + 'W_h', vmin = vmin, vmax = vmax)
    plot_weights_simple(W_h_aa, path + 'W_h_aa', 'From Ring Neuron', 'To Ring Neuron',
                        vmin = vmin, vmax = vmax)
    plot_weights_simple(W_h_ab_sorted, path + 'W_h_ab_sorted', vmin = vmin, vmax = vmax)
    plot_weights_simple(W_h_ba_sorted, path + 'W_h_ba_sorted', vmin = vmin, vmax = vmax)
    if not stationary:
        plot_weights_simple(W_i_b_sorted, path + 'W_i_b_sorted')

def plot_input_output_weights_stationary():
    input = np.eye(20)
    hzeros = np.zeros((20, 15))
    vzeros = np.zeros((15,20))
    hidden = np.random.uniform(-3,3, [35, 35])
    W_i_b = np.random.uniform(-3, 3, [2, 40])

    plot_weights_simple(hidden, path + 'W_h_random')
    plot_weights_simple(np.hstack((input, hzeros)), path + 'W_input')
    plot_weights_simple(np.vstack((input, vzeros)), path + 'W_output')
    plot_weights_simple(W_i_b, path +'W_i_b_initial')
    plot_weights_simple(np.array([[1, 0]]), path +'activity_velocity')
    plot_weights_simple(np.random.uniform(-3, 3, [30, 1]), path + 'activity_support')

if __name__ == '__main__':
    mpl.rcParams['font.size'] = 12

    # plot_input_output_weights_stationary()

    opts = utils.load_parameters('../test/stationary/parameters')
    opts.save_path = '../test/stationary/'
    sort_ix = sort_weights(opts)
    plot_stationary_weights(opts, sort_ix)

    # opts = utils.load_parameters('../test/non_stationary/parameters')
    # opts.save_path = '../test/non_stationary/'
    # sort_ix = sort_weights(opts)
    plot_nonstationary_weights(opts, sort_ix)