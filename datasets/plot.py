import numpy as np
from matplotlib import pyplot as plt
import os
from utils import adjust
import matplotlib as mpl

plt.style.use('default')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 7
mpl.rcParams['font.family'] = 'arial'

fig_path = '/Users/peterwang/Desktop/PYTHON/Neural_head_direction/_FIGURES'

def _easy_save(path, name, dpi=300, pdf=True):
    '''
    convenience function for saving figs while taking care of making folders
    :param path: save path
    :param name: save name
    :param dpi: save dpi for .png format
    :param pdf: boolean, save in another pdf or not
    :return:
    '''
    path = os.path.join(fig_path, path)
    os.makedirs(path, exist_ok=True)
    figname = os.path.join(path, name)
    print('figure saved in {}'.format(figname))
    plt.savefig(os.path.join(figname + '.png'), dpi=dpi)

    if pdf:
        plt.savefig(os.path.join(figname + '.pdf'), transparent=True)
    plt.close()

def _nice_figures(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(direction='out', length=1, width=.5)

def plot_stationary_inputs(inputs, labels, opts, n =3, path= 'example_input'):
    def helper(data, mode, xlabel = 'Angle', ylabel = 'Time'):
        fig = plt.figure(figsize=[3, 2])
        ax = fig.add_axes([.2, .2, .6, .6])
        plt.imshow(data, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.ylabel(ylabel)
        plt.yticks([-0.5, data.shape[0]-0.5], [0, data.shape[0]])
        plt.xlabel(xlabel)
        if mode in ['bump','one_hot']:
            plt.xticks([-.5, state_size//2, state_size - .5],[0, 180, 360])
        else:
            plt.xticks([])
        _nice_figures(ax)

    state_size = inputs.shape[-1] - 2
    for i in range(n):
        helper(inputs[i,:,:-2], opts.input_mode, xlabel='Angle', ylabel='Time')
        _easy_save(path, 'input_ring_{}'.format(i))

        helper(inputs[i, :, -2:], [], xlabel='AV', ylabel='Time')
        _easy_save(path, 'input_shift_{}'.format(i))

        helper(labels[i, :, :], opts.output_mode, xlabel='Angle', ylabel='Time')
        _easy_save(path, 'label_{}'.format(i))


def plot_moving_inputs(inputs, labels, opts):
    rc = (2,3)
    state = [x[:, :opts.state_size] for x in inputs[:rc[0]]]
    extra = [x[:, opts.state_size:opts.state_size+2] for x in inputs[:rc[0]]]

    i = 0
    # for bump labels
    if opts.output_mode == 'bump':
        fig, ax = plt.subplots(rc[0], rc[1])
        labels = labels[:rc[0]]
        for batch in zip(state, extra, labels):
            for d in batch:
                plot_ix = np.unravel_index(i, rc)
                cur_ax = ax[plot_ix]
                adjust(cur_ax)
                plt.sca(cur_ax)
                plt.imshow(d, cmap='RdBu_r', vmin=-1, vmax=1)
                cb = plt.colorbar()
                cb.set_ticks([-1, 1])
                i += 1
    else:
        f1, ax1 = plt.subplots(2,1)
        rad = np.arctan2(labels[:,:,1], labels[:,:,0])
        plt.sca(ax1[0])
        plt.plot(np.mod(rad[0], 2*np.pi))
        plt.sca(ax1[1])
        plt.imshow(inputs[0,:,-2:].T)

        r, c = 10, 20
        f2, ax2 = plt.subplots(r, c)
        for i in range(r):
            for j in range(c):
                ax2[i,j].plot(np.mod(rad[i*10+j], 2 * np.pi))

    plt.show()


def plot_nonneg(inputs, labels, opts):
    f1, ax1 = plt.subplots(2, 1)
    C = labels[:, :, 0] + labels[:, :, 1]
    S = labels[:, :, 2] + labels[:, :, 3]
    rad = np.arctan2(C, S)
    plt.sca(ax1[0])
    plt.plot(np.mod(rad[0], 2 * np.pi))
    plt.sca(ax1[1])
    plt.imshow(inputs[0, :, -2:].T)
    plt.show()


def plot_scalar_inputs(inputs, labels, vel_ipt, opts):
    rc = (2, 3)
    state_size = opts.state_size
    fig, ax = plt.subplots(rc[0], rc[1])
    state = [x[:, :state_size] for x in inputs[:rc[0]]]
    extra = [x[:, state_size:] for x in inputs[:rc[0]]]
    # labels = labels[:rc[0]]

    i = 0
    for batch in zip(state, extra, labels):
        for d in batch:
            if len(d.shape) == 1:
                d = d[:, np.newaxis]
            plot_ix = np.unravel_index(i, rc)
            cur_ax = ax[plot_ix]
            adjust(cur_ax)
            plt.sca(cur_ax)
            plt.imshow(d, cmap='RdBu_r', vmin=-1, vmax=1)
            cb = plt.colorbar()
            cb.set_ticks([-1, 1])
            i += 1

    state_size = opts.state_size
    vel_active = np.sum(vel_ipt, axis=2) > 0
    if state_size <= 72:
        ylim = [-1,1]
    else:
        ylim = [-2, 2]

    f1, ax1 = plt.subplots(10, 10)
    r, c = 0, 0
    while r < 10:
        ax1[r, c].plot(labels[r*10 + c])
        ax1[r, c].set_ylim(ylim)
        # vel = np.squeeze(np.argwhere(vel_active[r*10 + c]))
        # for v in vel:
        #     ax1[r, c].plot([v,v], ylim, c='gray', linestyle='--')

        c += 1
        if c >= 10:
            r += 1
            c = 0
    plt.show()