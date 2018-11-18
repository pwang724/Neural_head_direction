import os
import pickle as pkl
import numpy as np
from scipy.interpolate import griddata
import analysis.basics
import matplotlib.pyplot as plt

def get_activity(opts):
    state_size = opts.state_size
    save_path = opts.save_path
    activity_name = opts.activity_name

    with open(os.path.join(save_path, activity_name + '.pkl'), 'rb') as f:
        data_dict = pkl.load(f)

    states, predictions, labels = data_dict['states'], data_dict['predictions'], \
                                  data_dict['Y']

    noise_skip = 5
    states = np.stack(states, axis=1)  # examples x t x neurons, so each example is in axis 0
    states = states[:, noise_skip:, :]  # ignore the first time point where there is no label
    predictions = np.stack(predictions, axis=1)
    predictions = predictions[:, noise_skip:, :]
    labels = np.stack(labels, axis=0)
    labels = labels[:, noise_skip:, :]
    batch, time, n_rnn = states.shape
    _, _, n_state = labels.shape

    states = states.reshape(batch * time, n_rnn)  # each neuron's activity is a column
    pred = predictions.reshape(batch * time, -1)
    labels = labels.reshape(batch * time, n_state)

    # find the center of mass for each label

    # convert positions to angles in radians
    scale = 2 * np.pi / state_size
    x_rad = np.arange(state_size) * scale
    # convert angles to cartesian points
    cos, sin = np.cos(x_rad), np.sin(x_rad)
    cos_mean = np.sum(cos * labels, axis=1)
    sin_mean = np.sum(sin * labels, axis=1)
    com_rad = np.arctan2(sin_mean, cos_mean)
    com = (com_rad / scale) % state_size

    # find the velocity for each label
    vel = np.zeros_like(com)
    vel[1:] = com[1:] - com[:-1]
    vel[vel > opts.velocity_max + .5] -= state_size
    vel[vel < -(opts.velocity_max + .5)] += state_size
    vel[::time] *= 0  # velocity is zero at the start of each example

    # find the COM for the neurons
    pred_norm = np.sum(labels, axis=1)
    cos_mean = np.sum(cos * pred, axis=1) / pred_norm
    sin_mean = np.sum(sin * pred, axis=1) / pred_norm
    com_rad_n = np.arctan2(sin_mean, cos_mean)
    com_n = (com_rad_n / scale) % state_size

    com, vel, com_rad = com.ravel(), vel.ravel(), com_rad.ravel()
    points = np.around(np.stack([com, vel, com_rad, com_n], axis=1), 3)
    return points, states, labels


def plot_receptive_field(opts, points, activity, plot_stationary=False, save_name = None):
    """
    Plot the activity of a neuron using data from all processed batches.
    """
    sort_ix = analysis.basics.sort_weights(opts)
    activity[:,opts.state_size:] = activity[:,opts.state_size+sort_ix]

    x = np.arange(0, opts.state_size)
    # x = np.linspace(np.amin(points[:, 0]), np.amax(points[:, 0]))
    scale = 2 * np.pi / opts.state_size
    x_rad = x * scale
    cos, sin = np.cos(x_rad), np.sin(x_rad)
    if opts.velocity:
        y = np.linspace(np.amin(points[:, 1]), np.amax(points[:, 1]))
    else:
        y = np.zeros(1)

    x_mesh, y_mesh = np.meshgrid(x, y)
    cos, _ = np.meshgrid(cos, y)
    sin, _ = np.meshgrid(sin, y)
    if plot_stationary:
        nc, nr = 5, 4
        neurons = np.arange(opts.state_size)  # state neurons
    else:
        nc, nr = 7, 8
        neurons = np.arange(opts.state_size, opts.rnn_size)  # extra neurons

    f_linear, ax_linear = plt.subplots(ncols=nc, nrows=nr)
    for i, n in enumerate(neurons[:nc*nr]):
        plot_i = np.unravel_index(i, (nr, nc))
        z_lin = griddata(points[:, :2], activity[:, n], (x_mesh, y_mesh),
                         method='linear')
        plt.sca(ax_linear[plot_i])
        plt.contourf(x, y, z_lin, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.axis('off')

    if save_name is None:
        save_path = opts.save_path
        image_folder = opts.image_folder
        n = 'state' if plot_stationary else 'support'
        plot_name = os.path.join(save_path, image_folder, 'receptive_field_' + n + '.png')
    else:
        plot_name = os.path.join(save_name +  '.png')
    plt.savefig(plot_name, transparent=True, dpi=500)


# # find the global centroid
# if np.nanmax(z_lin) <= 0:
#     z_lin -= np.nanmean(z_lin)  # center activations at the median
#
# z_lin[np.isnan(z_lin)] = 0
# z_lin[z_lin < 0] = 0
# norm = np.sum(z_lin)
#
# cos_mean = np.sum(cos * z_lin) / norm
# sin_mean = np.sum(sin * z_lin) / norm
# com_rad = np.arctan2(sin_mean, cos_mean)
# com_x = (com_rad / scale) % 20
# com_y = np.sum(y_mesh * z_lin) / norm
# # plt.scatter(com_x, com_y, c='k')