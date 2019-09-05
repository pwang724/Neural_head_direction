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
    def helper(data, type, mode, xlabel = 'Angle', ylabel = 'Time'):
        fig = plt.figure(figsize=[3, 2])
        ax = fig.add_axes([.2, .2, .6, .6])
        plt.imshow(data, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.ylabel(ylabel)
        plt.yticks([-0.5, data.shape[0]-0.5], [0, data.shape[0]])
        plt.xlabel(xlabel)
        plt.axis('auto')
        if type == 'ring' or type == 'label':
            if mode in ['bump','onehot']:
                plt.xticks([-.5, state_size//2, state_size - .5],[0, 180, 360])
            elif mode == 'trig':
                if opts.non_negative_input:
                    plt.xticks([0, 1, 2, 3], ['+cos','-cos','+sin','-sin'])
                else:
                    plt.xticks([0, 1], ['cos', 'sin'])

        if type == 'AV':
            plt.xticks([0, 1], ['Left','Right'])
        _nice_figures(ax)

        rect_cb = [0.82, 0.2, 0.02, 0.6]
        ax = fig.add_axes(rect_cb)
        cb = plt.colorbar(cax=ax, ticks = [-1, 1])
        cb.outline.set_linewidth(0.5)
        cb.set_label('', fontsize=7, labelpad=-5)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('right')
        ax.tick_params(direction='out', length=1, width=.5)

    state_size = inputs.shape[-1] - 2
    for i in range(n):
        helper(inputs[i,:,:-2], type = 'ring', mode = opts.input_mode, xlabel='Angle', ylabel='Time')
        _easy_save(path, 'input_ring_{}'.format(i))

        helper(inputs[i, :, -2:], type = 'AV', mode = [], xlabel='AV', ylabel='Time')
        _easy_save(path, 'input_shift_{}'.format(i))

        helper(labels[i, :, :], type = 'label', mode = opts.output_mode, xlabel='Angle', ylabel='Time')
        _easy_save(path, 'label_{}'.format(i))