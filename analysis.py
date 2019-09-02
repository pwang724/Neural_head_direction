import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import utils
import os
import config
import pickle as pkl
from scipy.interpolate import griddata
import seaborn as sns
import pandas as pd
import tables
import re
from collections import defaultdict
import tf_train as train
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def radian_output(A, opts):
    if opts.nonneg_input:
        C_pred = A[:, :, 0] - A[:, :, 1]
        S_pred = A[:, :, 2] - A[:, :, 3]
        rad = np.arctan2(C_pred, S_pred)
    else:
        C_pred = A[:, :, 0]
        S_pred = A[:, :, 1]
        rad = np.arctan2(C_pred, S_pred)
    return rad


def get_com(A, opts, posn=None, axis=-1):
    """
    Get the center of mass over a given axis.
    :param posn:
    :param com:
    """
    if posn is None:
        posn = np.arange(A.shape[axis])

    norm = np.sum(A, axis=axis)
    norm[norm == 0] = 1
    if opts.linear_track:
        com = np.sum(posn * A, axis=axis) / norm
    else:  # find radial centers of mass
        if opts.output_mode == 'bump':
            posn = posn * 2 * np.pi / opts.state_size

        cos, sin = np.cos(posn), np.sin(posn)
        cos_mean = np.sum(cos * A, axis=axis)
        sin_mean = np.sum(sin * A, axis=axis)
        com = np.arctan2(sin_mean, cos_mean)
        if opts.output_mode == 'bump':
            com = np.mod(com, np.pi*2) * opts.state_size / (2 * np.pi)
    return com


def plot_performance(opts, eval=True, data=None, retrain=False):
    save_path = opts.save_path
    if retrain:
        train.train(opts)

    if eval:
        opts.activity_name = 'moving_activity'
        train.eval(opts, data)

    activity_name = 'moving_activity'
    with open(os.path.join(save_path, activity_name + '.pkl'), 'rb') as f:
        data_dict = pkl.load(f)
    plot_performance_helper(opts, data_dict)


def plot_performance_helper(opts, data_dict, name=''):
    save_path = opts.save_path
    image_folder = opts.image_folder
    output_mode = opts.output_mode

    plot_path = os.path.join(save_path, image_folder)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    states, predictions, ipt, labels = data_dict['states'], data_dict['predictions'], data_dict['X'], data_dict['Y']
    if name:
        name = ' ' + name
    if output_mode == 'trig':
        pred_rad, lab_rad = radian_output(predictions, opts), radian_output(labels, opts)
        tup = [('', [pred_rad[i], lab_rad[i]]) for i in range(200)]
        plot_name = os.path.join(plot_path, f'moving radian outputs' + name)
        ylim = [-np.pi - .1, np.pi + .1]
        utils.subplot_easy(tup, 10, 20, plot_name, orderC=False, ylim=ylim, ax_op=['tight'], hide_ticks=True)
        # utils.subplot_easy(tup, 10, 20, plot_name, tight_axes=True, orderC=False, ylim=ylim, hide_ticks=True)

    elif output_mode in ['scalar', 'bump']:
        if output_mode == 'bump':
            com = get_com(predictions, opts)
            labels = get_com(labels, opts)
            # com = np.argmax(predictions, axis=2)
            # labels = np.argmax(labels, axis=2)

        tup = [('', [np.squeeze(com[i,:]), np.squeeze(labels[i,:])]) for i in range(200)]
        plot_name = os.path.join(plot_path, f'moving output tracking' + '')
        utils.subplot_easy(tup, 10, 20, plot_name, orderC=False,  ax_op=['tight'], hide_ticks=True)
        # utils.subplot_easy(tup, 10, 20, plot_name, orderC=False, ylim=ylim, ax_op=['tight'], hide_ticks=True)
    plt.close('all')


def plot_loss(opts):
    save_path = opts.save_path
    image_folder = opts.image_folder
    log_name = opts.log_name
    with open(os.path.join(save_path, log_name + '.pkl'), 'rb') as f:
        log_dict = pkl.load(f)

    plot_path = os.path.join(save_path, image_folder)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    plot_name = os.path.join(save_path, image_folder, 'mse_loss.png')

    mse_loss = np.array(log_dict['mse_loss'])
    tup = [('MSE Loss', mse_loss)]
    utils.subplot_easy(tup, 1, 1, plot_name)


def interpolate_activity(avg_activity, opts, load_df=False, save_name=''):
    save_path = opts.save_path
    posn, vel, activity, xyz_data = avg_activity

    # Interpolate the activities for each neuron
    xmesh, ymesh = np.meshgrid(posn, vel)
    if load_df:
        with open(os.path.join(save_path, 'interp.pkl'), 'rb') as f:
            interpolated = pkl.load(f)
    else:
        print("Interpolating...\n")
        itp_activity = []
        for neuron in xyz_data:
            itp_activity.append(griddata(neuron[:, :2], neuron[:, 2], (xmesh, ymesh), method='linear'))
        interpolated = np.stack(itp_activity, axis=0)
        interpolated[np.isnan(itp_activity)] = 0
        if save_name:
            name = os.path.join(save_path, save_name + '_interp.pkl')
        else:
            name = 'interp.pkl'
        with open(os.path.join(save_path, name), 'wb') as f:
            pkl.dump(interpolated, f)
    return interpolated


def analyze_nonstationary_weights(opts, plot=True, eval=False, load_df=True, data=None):
    """Visualization of trained network."""
    state_size = opts.state_size
    rnn_size = opts.rnn_size
    save_path = opts.save_path
    image_folder = opts.image_folder
    weight_name = opts.weight_name
    output_mode = opts.output_mode
    input_mode = opts.input_mode
    type_string = f'{input_mode}-{output_mode}-{state_size}'
    # opts.subtrack = False
    EI_in = opts.EI_in
    EI_h = opts.EI_h
    EI_out = opts.EI_out

    if eval:
        train.eval(opts, data)

    plot_path = os.path.join(save_path, image_folder)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # load
    with open(os.path.join(save_path, weight_name + '.pkl'), 'rb') as f:
        weight_dict = pkl.load(f)

    W_in = weight_dict['model/input_weights:0']
    W_h = weight_dict['model/hidden_weights:0']
    W_out = weight_dict['model/output_weights:0']
    W_h_bias = weight_dict['model/hidden_bias:0']
    W_out_bias = weight_dict['model/output_bias:0']

    if EI_h or EI_out:
        prop_ex = opts.prop_ex
        nE = int(prop_ex * rnn_size)
        ei_mask = np.eye(rnn_size)
        ei_mask[nE:] *= -1

    if EI_in:
        W_in = np.abs(W_in)
    if EI_h:
        W_h = np.abs(W_h) * (1 - np.eye(rnn_size))
        W_h = np.dot(ei_mask, W_h)
        E_ix = np.arange(rnn_size) < nE
        I_ix = ~E_ix
    if EI_out:
        W_out = np.dot(ei_mask, np.abs(W_out))

    activity_name = 'moving_activity'
    with open(os.path.join(save_path, activity_name + '.pkl'), 'rb') as f:
        data_dict = pkl.load(f)

    avg_act = average_neural_activity(data_dict, opts, load_df)
    itp_activity = interpolate_activity(avg_act, opts, load_df)
    posn, vel, activity, data = avg_act

    # Interpolate the activities for each neuron
    # xmesh, ymesh = np.meshgrid(posn, vel)
    # if load_df:
    #     with open(os.path.join(save_path, 'interp.pkl'), 'rb') as f:
    #         itp_activity = pkl.load(f)
    # else:
    #     print("Interpolating...\n")
    #     itp_activity = []
    #     for neuron in data:
    #         itp_activity.append(griddata(neuron[:, :2], neuron[:, 2], (xmesh, ymesh), method='linear'))
    #     itp_activity = np.stack(itp_activity, axis=0)
    #     itp_activity[np.isnan(itp_activity)] = 0
    #     with open(os.path.join(save_path, 'interp.pkl'), 'wb') as f:
    #         pkl.dump(itp_activity, f)

    max_activity = np.amax(activity)
    max_avg_activity = np.amax(itp_activity)
    neuron_max_avg = np.amax(itp_activity, axis=(1,2))  # get the max over each neuron average
    active_thresh = 0
    # active_thresh = .05
    active_ix = neuron_max_avg > active_thresh
    # active_ix = neuron_max_avg > (max_avg_activity * active_thresh)
    inactive_ix = (1-active_ix).astype(np.bool)

    active_neurons = itp_activity[active_ix]
    neuron_category = np.zeros(rnn_size)

    # Sort the neurons by velocity responses
    # common thresholds: 1e-3 normally, 2e-1 for giant activities
    ring_threshold = 5e-4
    vel_category, templates, vel_slice = template_classify_neurons(vel, active_neurons,
                                                                   thresh=ring_threshold,
                                                                   filter_stationary=False,
                                                                   show_cov=False)
    neuron_category[active_ix] = vel_category
    neuron_category[inactive_ix] = -1
    # vel_category, templates = template_classify_neurons(vel, itp_activity)

    # Sort the neurons by position responses
    com_x, posn_slice = posn_classify_neurons(posn, itp_activity, opts, com=False)

    # sort the indices of the neurons
    # templates = [left_vel, right_vel, stationary, both_vel]

    category_ix = []
    E_category_ix = []
    I_category_ix = []
    for j in range(len(templates)+1):
        bool_ix = neuron_category == j
        ix = category_sort(bool_ix, com_x)
        category_ix.append(ix)

        if EI_h:
            E_cat = E_ix * bool_ix * active_ix
            ix = category_sort(E_cat, com_x)
            E_category_ix.append(ix)

            I_cat = I_ix * bool_ix * active_ix
            ix = category_sort(I_cat, com_x)
            I_category_ix.append(ix)

    inactive_ix = np.nonzero(active_ix==False)[0]
    category_ix.append(inactive_ix)
    sort_ix = np.concatenate(category_ix)
    com_x = com_x[sort_ix]

    Win_phase = W_in[:, sort_ix]
    Wh_phase = W_h[sort_ix, :][:, sort_ix]
    Wout_phase = W_out[sort_ix, :]

    useful_end = np.sum(active_ix)
    Wh_clean = Wh_phase[:useful_end, :useful_end]

    ring_length = len(category_ix[0])
    Wh_attractor = Wh_clean[:ring_length, :ring_length]
    Wh_velocity = Wh_clean.copy()
    Wh_velocity[:ring_length, :ring_length] = 0

    # standard indices
    category_str = ('ring', 'left_vel', 'right_vel', 'stationary', 'both_vel')
    ix_dict = {}
    for i, cat in enumerate(category_str):
        ix_dict[cat] = category_ix[i]
    ix_dict['inactive'] = inactive_ix
    ix_dict['sort_ix'] = sort_ix

    sorted_activity = itp_activity[sort_ix]
    vel_slice = np.amax(itp_activity, axis=2)
    posn_slice = np.amax(itp_activity, axis=1)

    print(f'Max activation: {max_activity}, Averaged maximum: {max_avg_activity}')
    print(f'{useful_end} useful, {rnn_size - useful_end} unused neurons')
    print(f'{len(category_ix[0])} position neurons, {len(category_ix[1])} left shift neurons, '
          f'{len(category_ix[2])} right shift neurons')

    if EI_h:
        E_sort = np.concatenate(E_category_ix)
        I_sort = np.concatenate(I_category_ix)
        EI_sort = np.concatenate([E_sort, I_sort, inactive_ix])

        Wh_EI = W_h[EI_sort, :][:, EI_sort]
        Wh_E = Wh_EI[:len(E_sort), :useful_end]
        Wh_I = Wh_EI[len(E_sort):useful_end, :useful_end]
        print(f'{np.sum(E_ix)} active excitatory, {np.sum(I_ix)} active inhibitory')

        # EI indices
        E_cat_str = ['E_' + cat for cat in category_str]
        for cat, cat_ix in zip(E_cat_str, E_category_ix):
            ix_dict[cat] = cat_ix

        I_cat_str = ['I_' + cat for cat in category_str]
        for cat, cat_ix in zip(I_cat_str, I_category_ix):
            ix_dict[cat] = cat_ix

        ix_dict['sorted_E_ix'] = E_ix[sort_ix]
        ix_dict['sorted_I_ix'] = I_ix[sort_ix]
        sort_len = np.cumsum([len(cat) for cat in E_category_ix] + [len(cat) for cat in I_category_ix])
        sort_cat = [cat for cat in E_cat_str] + [cat for cat in I_cat_str]

    with open(os.path.join(save_path, 'ix_dict.pkl'), 'wb') as f:
        pkl.dump(ix_dict, f)

    if plot:
        r = int(np.sqrt(rnn_size))
        c = r
        while r * c < rnn_size:
            c += 1

        plot_name = os.path.join(save_path, image_folder, f'Wh_sorted {type_string}')
        tup = [('Wh, Useful neurons', Wh_clean), ('Wh, Full', Wh_phase)]
        utils.subimage_easy(tup, 2, 1, plot_name, vmin=-.3, vmax=.3)

        tup = [('Win', Win_phase), ('Wh', Wh_phase), ('Wout', Wout_phase)]
        plot_name = os.path.join(save_path, image_folder, f'Weights sorted {type_string}')
        utils.subimage_easy(tup, 3, 1, plot_name)

        tup = [('Win', Win_phase[:, :useful_end]), ('Wh', Wh_clean), ('Wout', Wout_phase[:useful_end, :])]
        plot_name = os.path.join(save_path, image_folder, f'Active weights sorted {type_string}')
        utils.subimage_easy(tup, 3, 1, plot_name, vmin=-.5, vmax=.5)

        tup = [('Attractor', Wh_attractor), ('Shift', Wh_velocity)]
        plot_name = os.path.join(save_path, image_folder, f'separated weights {type_string}')
        utils.subimage_easy(tup, 2, 1, plot_name, vmin=-.3, vmax=.3)

        if opts.EI_h:
            # tup = [('Excitatory', Wh_E), ('Inhibitory', Wh_I)]
            # plot_name = os.path.join(save_path, image_folder, f'EI separated')
            # utils.subimage_easy(tup, 2, 1, plot_name, vmin=-.1, vmax=.1)

            tup = [('Excitatory-Inhibitory', Wh_EI[:useful_end,:useful_end])]
            plot_name = os.path.join(save_path, image_folder, f'EI')
            utils.subimage_easy(tup, 1, 1, plot_name, vmin=-.1, vmax=.1)

            # tup = [(cat, W_h[ix][:,sort_ix][:,:useful_end]) for cat, ix in zip(E_cat_str, E_category_ix)] + \
            #       [(cat, W_h[ix][:,sort_ix][:,:useful_end]) for cat, ix in zip(I_cat_str, I_category_ix)]
            # plot_name = os.path.join(save_path, image_folder, f'EI type separated')
            # utils.subimage_easy(tup, 2, 5, plot_name, vmin=-.1, vmax=.1, fontsize=8)

            # show how different sides connect to each other
            left_ix = np.concatenate([ix_dict['E_left_vel'], ix_dict['I_left_vel']], axis=0)
            right_ix = np.concatenate([ix_dict['E_right_vel'], ix_dict['I_right_vel']], axis=0)
            tup = [('Left connectivity', W_h[left_ix,:][:,left_ix]), ('Right connectivity', W_h[right_ix,:][:,right_ix])]
            plot_name = os.path.join(save_path, image_folder, f'EI_shifters')
            utils.subimage_easy(tup, 2, 1, plot_name, vmin=-.1, vmax=.1)

            f, ax = plt.subplots()
            # plt.imshow(Wh_EI, vmin=-.1, vmax=.1, cmap='RdBu_r')
            plt.imshow(Wh_EI[:useful_end, :useful_end], vmin=-.2, vmax=.2, cmap='RdBu_r')
            [ax.spines[loc].set_visible(False) for loc in ['bottom', 'top', 'left', 'right']]
            Lprev = 0
            for L in sort_len:
                if L-Lprev == 0:
                    continue
                plt.plot([-.5,useful_end-.5], [L-.5,L-.5], c='k', linewidth=.5)
                plt.plot([L-.5,L-.5], [-.5,useful_end-.5], c='k', linewidth=.5)
                Lprev = L
            plt.xlim([-.5,useful_end+.5])
            plt.ylim([-.5,useful_end+.5])
            ticks, ticklab = [], []
            tprev = 0
            for L, cat in zip(sort_len, sort_cat):
                if L-tprev == 0:
                    continue

                ticks.append(L-.5)
                ticklab.append(cat)
                tprev = L

            ticks.insert(0, -.5)
            aves = []
            for i, x in enumerate(ticks[1:],1):
                aves.append((ticks[i-1] + ticks[i]) / 2)

            plt.xticks(aves, ticklab, fontsize=4)
            plt.yticks(aves, ticklab, fontsize=4)
            plt.axis('image')
            ax.xaxis.set_tick_params(width=.5)
            ax.xaxis.set_ticks_position('top')
            ax.yaxis.set_tick_params(width=.5)
            ax.invert_yaxis()
            f.savefig(os.path.join(plot_path, 'Wh grid'), bbox_inches='tight', figsize=(14, 10), dpi=500)
            # plt.show()


        p_ylim = [0, max_activity]
        xlim = [posn[0], posn[-1]]
        ylim = [vel[0], vel[-1]]

        mesh = np.meshgrid(posn, vel)
        lim = (xlim, ylim, p_ylim)
        contour_plot(sorted_activity, mesh, lim, opts, thresh=ring_threshold)

        if opts.EI_h:
            # excitatory contour plot
            z_E = sorted_activity[E_ix[sort_ix]]
            if np.size(z_E):
                contour_plot(z_E, mesh, lim, opts, name='E')

            # inhibitory
            z_I = sorted_activity[I_ix[sort_ix]]
            if np.size(z_I):
                contour_plot(z_I, mesh, lim, opts, name='I')

    plt.close('all')
    return useful_end


def category_sort(bool_ix, com):
    phase = com[bool_ix]
    sort = np.argsort(phase)
    ix = np.flatnonzero(bool_ix)
    ix = ix[sort]
    return ix


def average_neural_activity(data_dict, opts, load_df=False, dfname=''):
    """Get the average activity for each neuron at each position and velocity."""
    rnn_size = opts.rnn_size
    save_path = opts.save_path
    time_loss_start = opts.time_loss_start
    time_loss_end = opts.time_loss_end
    output_mode = opts.output_mode
    states, ipt, labels, loss = data_dict['states'], data_dict['X'], data_dict['Y'], data_dict['loss']

    # Find the average response of each neuron for each position and velocity
    # posn, the location of each data point, using the labels (N x T)
    # vel, the velocity signal at each data point, using the inputs (N x T)
    # act, the activation of each neuron at each data point (N x T x D)

    # position - either an angle in radians or the center of an activity bump
    if output_mode == 'trig':
        lab_rad = radian_output(labels, opts)
        posn = lab_rad[:, time_loss_start:time_loss_end]
    else:  # output_mode == 'bump':
        # labels = np.argmax(labels, axis=2)
        labels = get_com(labels, opts)
        posn = np.squeeze(labels[:, time_loss_start:time_loss_end])

    # velocities: left turns (first ix) are negative, right turns (2nd ix) are positive
    vel = ipt[:, time_loss_start:, -2:]
    vel = vel[:, :, 1] - vel[:, :, 0]

    # activation
    act = states[:, time_loss_start:, :]  # (N x T x D)

    # find the average activity for each position/velocity combination using pandas
    posn = np.round(posn.ravel(), 4)  # N*T vector
    vel = np.round(vel.ravel(), 4)  # N*T vector
    act = act.reshape(-1, rnn_size)  # (N*T x D) array
    num_data_points, _ = act.shape

    # create labels (indices) for every entry of a pandas dataframe
    network_ix = np.arange(rnn_size)
    array_ix = ([[posn[ix]] * rnn_size,
                 [vel[ix]] * rnn_size,
                 network_ix,
                 act[ix]] for ix in range(num_data_points))  # index generator

    dfname = os.path.join(save_path, dfname + 'dataframe.h5')
    if load_df:
        df = pd.read_hdf(dfname, key='df')
    else:
        print('\nAnalyzing activity...')
        dfs = []
        names = ['position', 'vel_signal', 'neuron']
        for ix in array_ix:
            cur_act = ix[3]
            tuples = list(zip(ix[0], ix[1], ix[2]))
            index = pd.MultiIndex.from_tuples(tuples, names=names)
            dfs.append(pd.DataFrame(cur_act, index=index))

        df = pd.concat(dfs)
        df = df.groupby(names).mean()  # average across position/velocity combos for each neuron
        df.to_hdf(dfname, key='df', mode='w')

    posn = np.round(df.index.levels[0].values, 4)
    vel = df.index.levels[1].values
    neuron = df.index.levels[2]
    # get every posn/vel/activity combo for each neuron (D x N*T x 3)
    xyz_data = np.array([df.xs(nn, level='neuron').reset_index().values for nn in neuron])
    return posn, vel, act, xyz_data


def template_classify_neurons(velocity, activity, thresh=1e-3, filter_stationary=False, show_cov=False):
    """Sorting neurons using a set of templates."""
    middle = np.nonzero((velocity <= .5) * (velocity >= -.5))
    right = np.nonzero(velocity > 0)
    left = np.nonzero(velocity < 0)

    # make templates
    left_vel = np.zeros_like(velocity)
    left_vel[left] = np.abs(velocity[left])
    right_vel = np.zeros_like(velocity)
    right_vel[right] = velocity[right]
    ring = np.ones_like(velocity) + np.random.randn(velocity.shape[0]) * .01
    both_vel = np.abs(velocity)
    # stationary = 1 - both_vel
    stationary = np.zeros_like(velocity)
    stationary[middle] = 1 - np.abs(velocity[middle])
    # templates = [ring, left_vel, right_vel, stationary, both_vel]
    templates = [left_vel, right_vel, stationary, both_vel]
    ix_dict = dict(ring=0, left=1, right=2, stationary=3, both=4, inactive=-1)

    # get the correlation between a template and a flattened velocity contour
    # flatten interp
    vel_slice = np.amax(activity, axis=2)
    r = np.array([[np.corrcoef(v, temp)[0,1] for temp in templates] for v in vel_slice])
    r_category = np.argmax(r, axis=1) + 1
    cov = np.array([np.cov(v) for v in vel_slice])
    if show_cov:
        plt.plot(cov, 'o')
        plt.show()

    if filter_stationary:
        ring = (cov < thresh) * (r_category != ix_dict['stationary'])
        r_category[ring] = 0
    else:
        r_category[cov < thresh] = 0
        # r_category[r_category == ix_dict['stationary']] = 0
    return r_category, templates, vel_slice


def posn_classify_neurons(posn, activity, opts, reduction='max', com=False):
    """
    :param posn: a vector with all of the possible positions. (state_size)
    :param reduction: the method of reducing activity at each position. Max or mean.
    :param com: whether to take the activity center of mass (as before) or to take the maximal position.
    """
    if reduction == 'max':
        posn_slice = np.amax(activity, axis=1)  # (N x state_size) array, activity at each position
    else:
        posn_slice = np.mean(activity, axis=1)

    if com:
        com = get_com(posn_slice, opts)
    else:
        com = posn[np.argmax(posn_slice, axis=1)]
    return com, posn_slice


def contour_plot(z_interp, mesh, lim, opts, name='', posn_slice=False, thresh=None):
    rnn_size = z_interp.shape[0]
    xmesh, ymesh = mesh
    xlim, ylim, p_ylim = lim

    nr = np.ceil(np.sqrt(rnn_size)).astype(np.int32)
    nc = np.ceil(rnn_size / nr).astype(np.int32)
    c, r = 0, 0
    f_linear, ax_linear = plt.subplots(ncols=nc, nrows=nr)
    if posn_slice:
        f_p, ax_p = plt.subplots(ncols=nc, nrows=nr)

    if thresh:
        plt.suptitle(f'Ring threshold = {thresh}')

    # for z, cx, cy in zip(z_interp):  #, com_x, com_y):
    for z in z_interp:
        cur_ax = ax_linear[r, c]
        plt.sca(cur_ax)
        plt.contourf(xmesh, ymesh, z, cmap='RdBu_r', extend='both', vmin=0)
        # plt.contourf(xmesh, ymesh, z, cmap='RdBu_r', extend='both', vmin=0, vmax=.5)

        plt.xlim(xlim)
        plt.ylim(ylim)

        cur_ax.set_xticklabels([])
        cur_ax.set_yticklabels([])
        cur_ax.set_xticks([])
        cur_ax.set_yticks([])

        if posn_slice:
            cur_ax = ax_p[r, c]
            cur_ax.set_ylim(p_ylim)
            cur_ax.plot(xmesh[1], z[1])
            cur_ax.set_xticklabels([])
            cur_ax.set_yticklabels([])
            cur_ax.set_xticks([])
            cur_ax.set_yticks([])

        c += 1
        if c == nc:
            c = 0
            r += 1
        if r == nr:
            break

    save_path = opts.save_path
    image_folder = opts.image_folder
    if name:
        contour_name = name + '_activity_contour'
    else:
        contour_name = 'activity_contour'
    plot_name = os.path.join(save_path, image_folder, contour_name)
    f_linear.savefig(plot_name, bbox_inches='tight', figsize=(14, 10), dpi=500)
    if posn_slice:
        if name:
            slice_name = name + '_posn_slice'
        else:
            slice_name = 'posn_slice'
        plot_name = os.path.join(save_path, image_folder, slice_name)
        f_p.savefig(plot_name, bbox_inches='tight', figsize=(14, 10), dpi=500)
    plt.close('all')


def find_model(op_dict):
    # load parameter files, search for appropriate model
    modeldir = os.path.join(os.getcwd(), 'training')
    exp = re.compile('model([0-9]+)')
    found = []
    with os.scandir(modeldir) as models:
        for mod in models:
            dir = os.path.join(modeldir, mod.name)
            if not os.path.isdir(dir):
                continue
            if not os.listdir(dir):
                continue

            match = exp.match(mod.name)
            if match:
                fname = os.path.join(modeldir, mod.name, 'parameters')
                opts = utils.load_parameters(fname)
                m_dict = defaultdict(list)
                for k, v in vars(opts).items():
                    m_dict[k] = v
                # m_dict = vars(opts)
                m_property_match = np.array([m_dict[k] == v for k,v in op_dict.items()])
                if np.prod(m_property_match) == 1:  # all true
                    print(f'Model {match.group(1)} located in ' + opts.save_path)
                    found.append(opts)
    return found


def read_log(opts):
    save_path = opts.save_path
    with open(os.path.join(save_path, 'log.pkl'), 'rb') as f:
        data_dict = pkl.load(f)


def describe_model(model_ix, op_dict):
    modeldir = os.path.join(os.getcwd(), 'training')
    fname = os.path.join(modeldir, 'model' + str(model_ix).zfill(3), 'parameters')
    opts = utils.load_parameters(fname)
    m_dict = vars(opts)
    print(f"\nix: {model_ix}, Model attributes:")
    for k in op_dict.keys():
        print(f'{k}: {m_dict[k]}')


if __name__ == '__main__':
    op_dict = dict()
    # op_dict['state_size'] = 36
    # op_dict['velocity_min'] = 1
    # op_dict['velocity_max'] = 3
    # op_dict['EI_in'] = True
    # op_dict['EI_h'] = True
    # op_dict['EI_out'] = True
    # find_model(op_dict)

    root = './training/'

    ix = [3]
    # ix = [22,23,24]
    # for i in ix:
    #     describe_model(i, op_dict)

    # dirs = find_model(op_dict)
    dirs = [os.path.join(root, 'model' + str(n).zfill(3)) for n in ix]
    for d in dirs:
        fname = os.path.join(d, 'parameters')
        opts = utils.load_parameters(fname)
        # read_log(opts)

        # plot_activity(opts, retrain=True, eval=True)
        # analyze_nonstationary_weights(opts, plot=True, eval=True, load_df=False)

        # plot_activity(opts, eval=True)
        # analyze_nonstationary_weights(opts, plot=True, eval=False, load_df=False)

        # plot_activity(opts, eval=False)
        analyze_nonstationary_weights(opts, plot=True, eval=False, load_df=True)

        # analyze_nonstationary_weights(opts, plot=False, eval=False, load_df=True)

