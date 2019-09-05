import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import os
import utils

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import analysis
import experiments


def run_silencing(opts):
    # silencing_groups = ['ring']
    # silencing_groups = ['I_left_vel', 'I_right_vel']
    silencing_groups = ['E_left_vel', 'E_right_vel']
    for group in silencing_groups:
        experiments.silence(opts, group=group)


def analyze_experiment(opts):
    """Analyze for activity (make sure things were actually silent, maybe some noise
    Create an activity contour, plot performance"""
    # load silenced activity
    save_path = opts.save_path
    # silencing_groups = ['ring']
    silencing_groups = ['I_left_vel', 'I_right_vel']
    silencing_groups += ['E_left_vel', 'E_right_vel']
    silencing_groups = [gp + '_silence' for gp in silencing_groups]
    other_groups = ['ring_blast']

    # groups = other_groups
    groups = silencing_groups
    with open(os.path.join(save_path, 'ix_dict.pkl'), 'rb') as f:
        ix_dict = pkl.load(f)

    paths = [os.path.join(save_path, 'moving_activity.pkl')]
    for gp in groups:
        paths.append(os.path.join(save_path, gp + '.pkl'))

    activity_dict = []
    for p in paths:
        with open(p, 'rb') as f:
            act = pkl.load(f)
        activity_dict.append(act)

    # plot the performance
    # [analysis.plot_performance_helper(opts, a, b) for a, b in zip(activity_dict[1:], groups)]

    group_names = ['Normal'] + groups
    # [plot_trial_activity_separated(act, ix_dict, opts, t) for act, t in zip(activity_dict, group_names)]
    # plt.show()

    # plot_trial_performance(activity_dict, group_names, opts)

    plot_trial_current_separated(activity_dict[0], ix_dict, opts)#, 'excitatory ring current')

    # analyze the activity, maintaining the sorting order
    # average_act = [analysis.average_neural_activity(D, opts) for D in activity_dict]
    # itp_activity = [analysis.interpolate_activity(act, opts, save_name=gp) for act, gp in zip(average_act, silencing_groups)]
    # sorted_activity = [act[ix_dict['sort_ix']] for act in itp_activity]
    #
    # posn, vel, _, _ = average_act[0]
    # xlim = [posn[0], posn[-1]]
    # ylim = [vel[0], vel[-1]]
    #
    # mesh = np.meshgrid(posn, vel)
    # lim = (xlim, ylim, None)

    # for act, gp in zip(sorted_activity, silencing_groups):
    #     analysis.contour_plot(act, mesh, lim, opts, thresh=5e-4, name=gp)
    #
    #     if opts.EI_h:
    #         # excitatory contour plot
    #         z_E = act[ix_dict['sorted_E_ix']]
    #         if np.size(z_E):
    #             analysis.contour_plot(z_E, mesh, lim, opts, name=gp+'_E')
    #
    #         # inhibitory
    #         z_I = act[ix_dict['sorted_I_ix']]
    #         if np.size(z_I):
    #             analysis.contour_plot(z_I, mesh, lim, opts, name=gp+'_I')


def run_posn_blast(opts):
    experiments.posn_blast(opts)

# def analyze_posn_blast(opts):
#     save_path = opts.save_path
#     with open(os.path.join(save_path, 'ring_blast.pkl'), 'rb') as f:
#         data_dict = pkl.load(f)
#
#     states, predictions, ipt, labels = data_dict['states'], data_dict['predictions'], data_dict['X'], data_dict['Y']
#     paths = [os.path.join(save_path, 'moving_activity.pkl')]
#     for gp in silencing_groups:
#         paths.append(os.path.join(save_path, gp + '_silence.pkl'))
#
#     activity_dict = []
#     for p in paths:
#         with open(p, 'rb') as f:
#             act = pkl.load(f)
#         activity_dict.append(act)
#
#     # plot the performance
#     [analysis.plot_performance_helper(opts, a, b) for a, b in zip(activity_dict[1:], silencing_groups)]
#
#     [plot_trial_activity_separated(act, ix_dict, opts, t) for act, t in
#      zip(activity_dict, ['Normal'] + silencing_groups)]
#     plt.show()
#
#     pass
#

def plot_trial_activity_separated(data_dict, ix_dict, opts, title='', trial=0):
    plot_path = os.path.join(opts.save_path, opts.image_folder)

    states, predictions, ipt, labels = data_dict['states'], data_dict['predictions'], data_dict['X'], data_dict['Y']
    vmax = np.amax(states)
    Ering = states[trial,:,ix_dict['E_ring']].T
    Iring = states[trial,:,ix_dict['I_ring']].T
    EL = states[trial,:,ix_dict['E_left_vel']].T
    ER = states[trial,:,ix_dict['E_right_vel']].T
    IL = states[trial,:,ix_dict['I_left_vel']].T
    IR = states[trial,:,ix_dict['I_right_vel']].T

    f, ax = plt.subplots(1,7)
    ax[0].imshow(ipt[0], cmap='RdBu_r', vmin=0, vmax=1)
    ax[1].imshow(Ering, cmap='RdBu_r', vmin=0, vmax=vmax)
    ax[2].imshow(Iring, cmap='RdBu_r', vmin=0, vmax=vmax)
    ax[3].imshow(EL, cmap='RdBu_r', vmin=0, vmax=vmax)
    ax[4].imshow(ER, cmap='RdBu_r', vmin=0, vmax=vmax)
    ax[5].imshow(IL, cmap='RdBu_r', vmin=0, vmax=vmax)
    ax[6].imshow(IR, cmap='RdBu_r', vmin=0, vmax=vmax)

    ax[0].set_title('Input')
    ax[1].set_title('E ring')
    ax[2].set_title('I ring')
    ax[3].set_title('E left\nshift')
    ax[4].set_title('E right\nshift')
    ax[5].set_title('I left\nshift')
    ax[6].set_title('I right\nshift')
    if title:
        plt.suptitle(title)

    # plt.tight_layout()
    f.savefig(os.path.join(plot_path, title+' activity.pdf'), format='pdf', bbox_inches='tight')
    plt.close()


def plot_trial_performance(data_dicts, group_names, opts, trial=0):
    save_path = opts.save_path
    image_folder = opts.image_folder
    output_mode = opts.output_mode

    plot_path = os.path.join(save_path, image_folder)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    f, ax = plt.subplots(1,len(group_names))
    for d, g, a in zip(data_dicts, group_names, ax):
        predictions, labels = d['predictions'], d['Y']
        T = labels.shape[1]

        if output_mode == 'trig':
            pred = analysis.radian_output(predictions, opts)[trial]
            lab = analysis.radian_output(labels, opts)[trial]

        elif output_mode == 'bump':
            pred = analysis.get_com(predictions, opts)[trial]
            lab = analysis.get_com(labels, opts)[trial]

        a.plot(pred, np.arange(T))
        a.plot(lab, np.arange(T))
        a.invert_yaxis()
        a.set_title(g, fontsize=8)
    # plt.axis('tight')
    plt.tight_layout()
    plot_name = os.path.join(plot_path, f'lesioned output tracking.pdf')
    f.savefig(plot_name, format='pdf', bbox_inches='tight')
    # utils.subplot_easy(tup, len(group_names), 1, plot_name, orderC=False, ax_op=['tight'], hide_ticks=True)
    plt.close('all')


def hide_ticks(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

def plot_trial_current_separated(data_dict, ix_dict, opts, title='', trial=0):
    EI_in = opts.EI_in
    EI_h = opts.EI_h
    EI_out = opts.EI_out
    rnn_size = opts.rnn_size
    plot_path = os.path.join(opts.save_path, opts.image_folder)

    # load
    with open(os.path.join(opts.save_path, opts.weight_name + '.pkl'), 'rb') as f:
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

    states, predictions, ipt, labels = data_dict['states'], data_dict['predictions'], data_dict['X'], data_dict['Y']
    vmax = np.amax(states)

    # get the current to each neuron from the excitatory and inhibitory neurons onto the ring
    full_current = np.zeros_like(states)
    ER_current = np.zeros_like(states)
    EL_current = np.zeros_like(states)
    IR_current = np.zeros_like(states)
    IL_current = np.zeros_like(states)

    Ering_ix = ix_dict['E_ring']
    EL_ix = ix_dict['E_left_vel']
    ER_ix = ix_dict['E_right_vel']
    IL_ix = ix_dict['I_left_vel']
    IR_ix = ix_dict['I_right_vel']

    for i in range(states.shape[1]):
        full_current[:,i,:] = np.dot(states[:,i,:], W_h) + W_h_bias
        EL_current[:,i,:] = np.dot(states[:,i,EL_ix], W_h[EL_ix]) + W_h_bias
        ER_current[:,i,:] = np.dot(states[:,i,ER_ix], W_h[ER_ix]) + W_h_bias
        IL_current[:,i,:] = np.dot(states[:,i,IL_ix], W_h[IL_ix]) + W_h_bias
        IR_current[:,i,:] = np.dot(states[:,i,IR_ix], W_h[IR_ix]) + W_h_bias

    cmin = np.amin(full_current)
    print(cmin)

    f, ax = plt.subplots(1,8)
    ax[0].imshow(states[trial, :, Ering_ix].T, cmap='RdBu_r', vmin=0, vmax=vmax)
    ax[1].imshow(full_current[trial, :, Ering_ix].T, cmap='RdBu_r', vmin=0, vmax=vmax)
    ax[2].imshow(EL_current[trial, :, Ering_ix].T, cmap='RdBu_r', vmin=0, vmax=vmax)
    ax[3].imshow(ER_current[trial, :, Ering_ix].T, cmap='RdBu_r', vmin=0, vmax=vmax)
    ax[4].imshow(np.abs(IL_current[trial, :, Ering_ix].T), cmap='RdBu_r', vmin=0, vmax=vmax)
    ax[5].imshow(np.abs(IR_current[trial, :, Ering_ix].T), cmap='RdBu_r', vmin=0, vmax=vmax)
    ax[6].imshow(EL_current[trial, :, Ering_ix].T + IL_current[trial, :, Ering_ix].T, cmap='RdBu_r', vmin=0, vmax=vmax)
    ax[7].imshow(ER_current[trial, :, Ering_ix].T + IR_current[trial, :, Ering_ix].T, cmap='RdBu_r', vmin=0, vmax=vmax)

    ax[0].set_title('Full activation', fontsize=6)
    ax[1].set_title('Full current', fontsize=6)
    ax[2].set_title('EL current', fontsize=6)
    ax[3].set_title('ER current', fontsize=6)
    ax[4].set_title('IL current', fontsize=6)
    ax[5].set_title('IR current', fontsize=6)
    ax[6].set_title('L current', fontsize=6)
    ax[7].set_title('R current', fontsize=6)
    [hide_ticks(a) for a in ax]
    if title:
        plt.suptitle(title)

    # plt.tight_layout()
    f.savefig(os.path.join(plot_path, 'excitatory ring current.pdf'), format='pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    root = './training/'
    ix = [3]
    dirs = [os.path.join(root, 'model' + str(n).zfill(3)) for n in ix]
    for d in dirs:
        fname = os.path.join(d, 'parameters')
        opts = utils.load_parameters(fname)
        # run_silencing(opts)
        analyze_experiment(opts)

        # run_posn_blast(opts)

