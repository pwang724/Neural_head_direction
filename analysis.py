import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import utils
import os
import pickle as pkl
import json
from scipy.interpolate import griddata
import pandas as pd
import re
from collections import defaultdict
import torch_train
import torch_model
from tools import torch2numpy


def get_weights(net, opts):
    weight_dict = defaultdict()
    for name, param in net.named_parameters():
        weight_dict[name] = torch2numpy(param)

    if opts.EI:
        ei_mask = weight_dict['ei_mask']
        weight_dict['h_w'] = np.matmul(ei_mask, np.abs(weight_dict['h_w']))
        weight_dict['i2h'] = np.abs(weight_dict['i2h'])
        weight_dict['h2o_w'] = np.matmul(ei_mask, np.abs(weight_dict['h2o_w']))

    return weight_dict

def get_data(opts, eval):
    fname = os.path.join(opts.save_path, 'test_log.pkl')
    if eval or not os.path.exists(fname):
        logger = torch_train.evaluate(opts, log=True)
    else:
        with open(fname, 'rb') as f:
            logger = pkl.load(f)
    return logger

def get_radian_outputs(data, opts):
    # covert all outputs to radians
    y, y_out = data['y'], data['y_out']
    # if y.shape[-1] == 2:  # sin/cos
    if opts.output_mode == 'trig':
        if opts.non_negative_output:
            C = y[:, :, 0] - y[:, :, 1]
            S = y[:, :, 2] - y[:, :, 3]
            yrad = np.arctan2(S, C)

            C = y_out[:, :, 0] - y_out[:, :, 1]
            S = y_out[:, :, 2] - y_out[:, :, 3]
            yout_rad = np.arctan2(S, C)
        else:
            yrad = np.arctan2(y[:, :, 1], y[:, :, 0])
            yout_rad = np.arctan2(y_out[:, :, 1], y_out[:, :, 0])

    else:  # bump or onehot
        yrad = get_radial_com(y, opts)
        yout_rad = get_radial_com(y_out, opts)

    data['yrad'] = yrad + np.pi
    data['yout_rad'] = yout_rad + np.pi
    return data

def get_radial_com(A, opts):
    """Get the radial center of mass over a given axis."""
    posn = np.arange(opts.state_size)
    posn = posn * 2 * np.pi / opts.state_size
    norm = np.sum(posn)

    cos, sin = np.cos(posn), np.sin(posn)
    cos_mean = np.sum(cos * A, axis=-1) / norm
    sin_mean = np.sum(sin * A, axis=-1) / norm
    com = np.arctan2(sin_mean, cos_mean)
    return com

def plot_performance(data_dict, plot_path):
    yout_rad, yrad = data_dict['yout_rad'], data_dict['yrad']
    data = [(yout_rad[i], yrad[i]) for i in range(100)]
    plot_name = 'performance'
    utils.subplot_easy(data, 10, plot_path, plot_name, ax_op=['tight'], tight_layout=True, linewidth=.5, hide_ticks=True)

def average_neural_activity(data, opts):
    """Get the average activity for each neuron at each position and velocity."""
    # position - an angle in radians using the labels (N x T)
    posn = data['yrad'][:, opts.time_loss_start:opts.time_loss_end]

    # velocities (N x T): left turns (first ix) are negative, right turns (2nd ix) are positive
    if opts.velocity_onehot:
        vel = data['x'][:, opts.time_loss_start:opts.time_loss_end, -2:]
        vel = vel[:, :, 1] - vel[:, :, 0]
    else:
        vel = data['x'][:, opts.time_loss_start:opts.time_loss_end, -1:]

    # act, the activation of each neuron at each data point (N x T x D)
    act = data['h'][:, opts.time_loss_start:opts.time_loss_end, :]

    # find the average activity for each position/velocity combination using pandas
    position = np.round(posn.ravel(), 4)  # N*T vector
    velocity = np.round(vel.ravel(), 4)  # N*T vector
    activity = act.reshape(-1, opts.rnn_size)  # (N*T x D) array

    if not opts.discrete:  # place positions into bins
        n_posn_bins = 36
        n_vel_bins = 20

        n_posn_interval = (2 * np.pi) / n_posn_bins
        n_vel_interval = 2 / n_vel_bins  # 2 is the interval from -1 to 1, div by nbins

        position = np.round(position / n_posn_interval) * n_posn_interval
        # position = np.mod(position, 2 * np.pi)
        velocity = np.round(velocity / n_vel_interval) * n_vel_interval

    xyz_ungrouped = np.concatenate([position[:, np.newaxis], velocity[:, np.newaxis], activity], axis=1)
    ix = ['posn', 'vel'] + np.arange(opts.rnn_size).tolist()
    df = pd.DataFrame(xyz_ungrouped, columns=ix)
    df = df.groupby(['posn','vel']).mean().reset_index()
    xyz = np.round(df.values, 4)
    return xyz[:,0], xyz[:,1], xyz[:,2:]

def interpolate_activity(position, velocity, activity):
    """
    Interpolate the activity of each neuron over every position and velocity combination.
    :return interpolated: (rnn_size x n_velocities x n_positions) array
    """
    xy = np.stack([position, velocity], axis=1)
    posn_set, vel_set = np.sort(list(set(position))), np.sort(list(set(velocity)))
    xmesh, ymesh = np.meshgrid(posn_set, vel_set)

    print("Interpolating...")
    itp_activity = []
    for i in range(activity.shape[1]):
        itp_activity.append(griddata(xy, activity[:,i], (xmesh, ymesh), method='linear'))
    interpolated = np.stack(itp_activity, axis=0)
    interpolated[np.isnan(itp_activity)] = 0

    print("Interpolation done\n")
    return interpolated, posn_set, vel_set, (xmesh, ymesh)

def template_classify_neurons(vel_set, itp_activity, opts, thresh=1e-3, show_cov=False, combine_stationary=False):
    """
    Sorting neurons using a set of templates.
    Categories: ring = 0, left = 1, right = 2, stationary = 3, both = 4, inactive = -1)
    """
    middle = (-.5 <= vel_set) & (vel_set <= .5)
    right = vel_set > 0
    left = vel_set < 0

    # make templates
    left_vel = np.zeros_like(vel_set)
    right_vel = np.zeros_like(vel_set)
    stationary = np.zeros_like(vel_set)

    left_vel[left] = np.abs(vel_set[left])
    right_vel[right] = vel_set[right]
    # stationary = 1 - np.abs(vel_set)
    stationary[middle] = 1 - np.abs(vel_set[middle])
    both_vel = np.abs(vel_set)
    templates = [left_vel, right_vel, stationary, both_vel]

    # get the correlation between a template and velocity tuning
    vel_slice = np.amax(itp_activity, axis=2)  # max activity at each velocity, (rnn_size x n_vel)
    r = np.array([[np.corrcoef(v, temp)[0,1] for temp in templates] for v in vel_slice])
    cov = np.array([np.cov(v) for v in vel_slice])
    if show_cov:  # use to set a threshold
        plt.plot(cov, 'o')
        plt.plot([0, len(cov)],[thresh, thresh], color='k', linestyle='--')
        plt.title('Covariance')
        plt.show()
        plt.close()

    category = np.argmax(r, axis=1) + 1
    if not opts.constrained:
        category[cov < thresh] = 0
        if combine_stationary:
            category[category == 3] = 0

    for i, cat in enumerate(('ring', 'left', 'right', 'stationary', 'both')):
        print(f'{cat}: {np.sum(category == i)}')

    return category, templates, vel_slice

def posn_sort_neurons(itp_activity, opts, reduce_max=True, com=False):
    """
    :param reduction: the method of summarizing activity at each position. Max or mean.
    :param com: whether to take the activity center of mass or to take the maximal position.
    """
    if reduce_max:
        posn_slice = np.amax(itp_activity, axis=1)
    else:
        posn_slice = np.mean(itp_activity, axis=1)

    if com:
        com = get_radial_com(posn_slice, opts)
    else:
        com = np.argmax(posn_slice, axis=1) * 2 * np.pi / opts.state_size
    return com, posn_slice

def sort_simple_network(category, posn_center, opts):
    ind = np.lexsort([posn_center, category])
    cat_sort = category[ind]
    category_ix = []
    for j in range(5):
        category_ix.append(ind[cat_sort == j])

    sort_ix = np.concatenate(category_ix)
    inactive_ix = np.flatnonzero(category == -1)
    category_ix.append(inactive_ix)
    full_sort_ix = np.concatenate([sort_ix, inactive_ix])

    # standard indices
    category_str = ('ring', 'left_vel', 'right_vel', 'stationary', 'both_vel')
    ix_dict = {}
    for i, cat in enumerate(category_str):
        ix_dict[cat] = category_ix[i]

    ix_dict['inactive'] = inactive_ix
    ix_dict['sort_ix'] = sort_ix
    ix_dict['full_sort_ix'] = full_sort_ix

    if opts.EI:
        E_category_ix = []
        nE = int(opts.rnn_size * opts.prop_ex)
        E_ix = np.arange(opts.rnn_size) < nE
        E_cat = category[E_ix]
        E_ind = np.lexsort([posn_center[E_ix], category[E_ix]])
        E_cat_sort = E_cat[E_ind]

        I_category_ix = []
        I_ix = ~E_ix
        I_cat = category[I_ix]
        I_ind = np.lexsort([posn_center[I_ix], category[I_ix]])
        I_cat_sort = I_cat[I_ind]
        for i, cat in enumerate(category_str):
            _E = E_ind[E_cat_sort == i]
            _I = I_ind[I_cat_sort == i] + nE

            E_category_ix.append(_E)
            I_category_ix.append(_I)
            ix_dict['E_' + cat] = _E
            ix_dict['I_' + cat] = _I

        ix_dict['E_sort_ix'] = np.concatenate(E_category_ix)
        ix_dict['I_sort_ix'] = np.concatenate(I_category_ix)

        E_inactive_ix = np.flatnonzero(E_cat == -1)
        E_category_ix.append(inactive_ix)
        E_full_sort_ix = np.concatenate([ix_dict['E_sort_ix'], E_inactive_ix])
        ix_dict['E_full_sort_ix'] = E_full_sort_ix

        I_inactive_ix = np.flatnonzero(I_cat == -1)
        I_category_ix.append(inactive_ix)
        I_full_sort_ix = np.concatenate([ix_dict['I_sort_ix'], I_inactive_ix])
        ix_dict['I_full_sort_ix'] = I_full_sort_ix

    with open(os.path.join(save_path, 'ix_dict.pkl'), 'wb') as f:
        pkl.dump(ix_dict, f)

    return ix_dict

def plot_simple_weights(weight_dict, ix_dict, plot_path, opts):
    ring_ix = ix_dict['ring']
    shift_ix = np.concatenate([ix_dict['left_vel'], ix_dict['right_vel']])
    core_ix = np.concatenate([ring_ix, shift_ix])
    non_ring_ix = np.concatenate([shift_ix, ix_dict['stationary'], ix_dict['both_vel']])
    sort_ix = ix_dict['sort_ix']

    # Note that pytorch weights within Linear modules are to-from, so they must be inverted
    if opts.EI:
        i2h = weight_dict['i2h']
        h2o = weight_dict['h2o_w']
        i2h_sort = i2h[:, sort_ix]
    elif opts.constrained:
        pos2attr = weight_dict['position2attractor.weight'].T
        vel2shift = weight_dict['velocity2shift.weight'].T
        h2o = weight_dict['h2o.weight'].T

        pos_ipt_sort = pos2attr[:, ring_ix]
        vel_ipt_sort = vel2shift[:, non_ring_ix-opts.state_size]
        i2h_sort = np.zeros([pos_ipt_sort.shape[0] + vel_ipt_sort.shape[0],
                             pos_ipt_sort.shape[1] + vel_ipt_sort.shape[1]])
        i2h_sort[:pos_ipt_sort.shape[0], :pos_ipt_sort.shape[1]] = pos_ipt_sort
        i2h_sort[pos_ipt_sort.shape[0]:, pos_ipt_sort.shape[1]:] = vel_ipt_sort
    else:
        i2h = weight_dict['i2h.weight'].T
        h2o = weight_dict['h2o.weight'].T
        i2h_sort = i2h[:, sort_ix]

    h_w = weight_dict['h_w']
    hw_sort = h_w[sort_ix, :][:, sort_ix]
    h2o_sort = h2o[sort_ix, :]

    # get the ring, then ring to shift, then shift to ring
    ring = h_w[ring_ix,:][:,ring_ix]
    r2s = h_w[ring_ix,:][:,shift_ix]
    s2r = h_w[shift_ix,:][:,ring_ix]

    # make plots, worry about EI later
    data = (i2h_sort, hw_sort, h2o_sort)
    subtitles = ('Win', 'Wh', 'Wout')
    utils.subimage_easy(data, 3, plot_path, 'weights_sorted', subtitles, vmin=-.3, vmax=.3)
    utils.subimage_easy([ring], 1, plot_path, 'ring', vmin=-.3, vmax=.3)
    utils.subimage_easy([r2s], 1, plot_path, 'ring_to_shifters', vmin=-.3, vmax=.3)
    utils.subimage_easy([s2r], 1, plot_path, 'shifters_to_ring', vmin=-.3, vmax=.3)

    if opts.EI:
        E_shift_ix = np.concatenate([ix_dict['E_left_vel'], ix_dict['E_right_vel']])
        I_shift_ix = np.concatenate([ix_dict['I_left_vel'], ix_dict['I_right_vel']])

        hw_E = h_w[ix_dict['E_sort_ix'],:][:,ix_dict['E_sort_ix']]
        hw_I = h_w[ix_dict['I_sort_ix'],:][:,ix_dict['I_sort_ix']]
        data = (hw_E, hw_I)
        subtitles = ('Excitatory Weights', 'Inhibitory Weights')
        utils.subimage_easy(data, 2, plot_path, 'EI_weights', subtitles, vmin=-.3, vmax=.3)

        E_ring = h_w[ix_dict['E_ring'], :][:, ix_dict['E_ring']]
        I_ring = h_w[ix_dict['I_ring'], :][:, ix_dict['I_ring']]
        E_r2s = h_w[ix_dict['E_ring'], :][:, E_shift_ix]
        I_r2s = h_w[ix_dict['I_ring'], :][:, I_shift_ix]
        E_s2r = h_w[E_shift_ix, :][:, ix_dict['E_ring']]
        I_s2r = h_w[I_shift_ix, :][:, ix_dict['I_ring']]

        # make plots, worry about EI later
        data = (E_ring, I_ring)
        utils.subimage_easy(data, 2, plot_path, 'EI_ring', vmin=-.3, vmax=.3)
        data = (E_r2s, I_r2s)
        utils.subimage_easy(data, 2, plot_path, 'EI_ring_to_shifters', vmin=-.3, vmax=.3)
        data = (E_s2r, I_s2r)
        utils.subimage_easy(data, 2, plot_path, 'EI_shifters_to_ring', vmin=-.3, vmax=.3)

def contour_plot(itp_activity, mesh, plot_path, nc=None, name='', posn_slice=False, thresh=None):
    """Make contour plots of each neuron's position and velocity tuning."""
    # sort the activities
    n_units = itp_activity.shape[0]
    if nc is None:
        nc = np.ceil(np.sqrt(n_units)).astype(np.int32)
    nr = np.ceil(n_units / nc).astype(int)
    f_contour, ax_contour = plt.subplots(ncols=nc, nrows=nr)
    axes = np.ravel(ax_contour)

    if thresh:
        plt.suptitle(f'Ring threshold = {thresh}')

    for zmesh, cur_ax in zip(itp_activity, axes):
        plt.sca(cur_ax)
        plt.contourf(mesh[0], mesh[1], zmesh, cmap='RdBu_r', extend='both', vmin=0)
        # plt.imshow(zmesh, vmin=0, aspect='auto')#, cmap='RdBu_r')

        utils.hide_axis_ticks(cur_ax)

    if name:
        name = name + '_'
    contour_name = name + 'activity_contour'
    plot_name = os.path.join(plot_path, contour_name)
    f_contour.savefig(plot_name, bbox_inches='tight', figsize=(3, 2), dpi=500)

    if posn_slice:
        data = [np.amax(z, axis=0) for z in itp_activity]
        utils.subplot_easy(data, nc, plot_path, name + 'posn_slice')

    plt.close('all')

def analyze_simple_network(opts, eval=False):
    plot_path = os.path.join('./_FIGURES', os.path.split(opts.save_path)[-1])

    opts, data_loader, net = torch_train._initialize(opts, reload=True, set_seed=False)
    weight_dict = get_weights(net, opts)
    data = get_data(opts, eval)
    data = get_radian_outputs(data, opts)  # represent output data as angles
    plot_performance(data, plot_path)

    # get the average activity of each neuron at each speed and position
    position, velocity, activity = average_neural_activity(data, opts)
    itp_activity, posn_set, vel_set, mesh = interpolate_activity(position, velocity, activity)

    neuron_max_avg = np.amax(activity, axis=0)  # get the max over each neuron average
    # active_thresh = 0
    active_thresh = .05
    active_ix = neuron_max_avg > active_thresh

    # plt.plot(neuron_max_avg, 'o')
    # plt.title('Max neural activities')
    # plt.plot([0, len(neuron_max_avg)], [active_thresh, active_thresh], color='k', linestyle='--')
    # plt.title('Activity')
    # plt.show()

    # classify the neurons by velocity tuning
    active_neurons = itp_activity[active_ix]
    category = np.zeros(opts.rnn_size)
    category[~active_ix] = -1

    # sorting options
    # common thresholds: 1e-3, 5e-4, 2e-1 for giant activities
    ring_threshold = 1e-3
    combine_stationary = True
    if opts.constrained:
        ring_ix = np.arange(opts.rnn_size) < opts.state_size
        active_ring = active_ix & ring_ix
        active_shift = active_ix & ~ring_ix
        ring = itp_activity[active_ring]
        shifters = itp_activity[active_shift]
        template_category, templates, vel_slice = template_classify_neurons(vel_set, shifters, opts,
                                                                            combine_stationary=True,
                                                                            show_cov=False)
        category[active_ring] = 0
        category[active_shift] = template_category
    else:
        template_category, templates, vel_slice = template_classify_neurons(vel_set, active_neurons, opts,
                                                                        thresh=ring_threshold,
                                                                        combine_stationary=combine_stationary,
                                                                        show_cov=True)
        category[active_ix] = template_category

    # Sort the neurons by position tuning
    posn_center, posn_slice = posn_sort_neurons(itp_activity, opts, com=False)

    # sort the indices of the neurons, then make plots
    ix_dict = sort_simple_network(category, posn_center, opts)
    plot_simple_weights(weight_dict, ix_dict, plot_path, opts)
    contour_plot(itp_activity[ix_dict['full_sort_ix']], mesh, plot_path)
    if opts.EI:
        contour_plot(itp_activity[ix_dict['E_full_sort_ix']], mesh, plot_path, name='E', nc=10)
        contour_plot(itp_activity[ix_dict['I_full_sort_ix']], mesh, plot_path, name='I')

    sort_settings = dict(ring_threshold=ring_threshold, constrained_ring=opts.constrained,
                         active_threshold=active_thresh, combine_stationary=combine_stationary)
    with open(os.path.join(plot_path, 'sort_settings.txt'), "w") as f:
        for k, v in sort_settings.items():
            f.write(str(k) + ' >>> ' + str(v) + '\n\n')


if __name__ == '__main__':
    save_path = './_DATA/trig_trig/'
    opts = torch_model.load_config(save_path)
    analyze_simple_network(opts)
