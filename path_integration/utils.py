import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl


def save_weights(weights, path, save_name):
    assert save_name is not None, "Please choose a name for the weights"
    save_name = path + '/' + save_name
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    weight_dict = {'W_ah': weights[0], 'W_sh': weights[1], 'W_hh': weights[2], 'W_out': weights[3], "bias": weights[4]}
    with open(save_name, 'wb+') as f:
        pkl.dump(weight_dict, f)

def load_weights(path, load_name):
    file = path + "/" + load_name
    with open(file, 'rb') as f:
        weight_dict = pkl.load(f)
        W_ah = weight_dict["W_ah"]
        W_sh = weight_dict["W_sh"]
        W_hh = weight_dict["W_hh"]
        W_out = weight_dict["W_out"]
        bias = weight_dict["bias"]

    return W_ah, W_sh, W_hh, W_out, bias

def pretty_image(tup, col, row, save_name, cbar=False):
        """input: list of tuples
        first arg of each tuple: title
        second arg of each tuple: matrix to plot
        """
        c, r = 0, 0
        fig, ax = plt.subplots(nrows=row, ncols=col)
        for t, w in tup:
            if row > 1 and col > 1:
                cur_axis = ax[r, c]
            elif row > 1 or col > 1:
                cur_axis = ax[c]
            else:
                cur_axis = ax

            plt.sca(cur_axis)
            sns.heatmap(w, cmap='RdBu_r', ax=ax[r, c], cbar=cbar, center=0)
            # sns.heatmap(w, cmap='RdBu_r', vmin=-1, vmax=1, ax=ax[r, c], cbar=False, center=0)
            if t != '':
                cur_axis.set_title(t)
            cur_axis.axis('off')
            cur_axis.axis('image')

            c += 1
            if c >= col:
                r += 1
                c = 0

        plt.tight_layout()
        fig.savefig(save_name, bbox_inches='tight', figsize = (14,10), dpi=300)
        plt.close()

def multiplot(ax, mat):
    try:
        if len(np.squeeze(mat).shape) == 1:
            ax.plot(mat)
        else:
            ax.plot(mat[:, 0], mat[:, 1])
    except TypeError:
        print("Plotting error - pass the data to be plotted as a numpy array or as a list of numpy arrays.")

def pretty_plot(tup, col, row, save_name, xlim=None, ylim=None, hide_ticks=False, ax_op=None, suptitle=None):
    """input: list of tuples
    first arg of each tuple: title
    second arg of each tuple: matrix to plot
    """
    c, r = 0, 0
    fig, ax = plt.subplots(nrows=row, ncols=col)
    if suptitle:
        plt.suptitle(suptitle)

    for t, w in tup:
        if row > 1 and col > 1:
            cur_axis = ax[r, c]
        elif row > 1 or col > 1:
            cur_axis = ax[c]
        else:
            cur_axis = ax

        if isinstance(w, list) or isinstance(w, tuple):
            # multiple data for each plot
            for mat in w:
                multiplot(cur_axis, mat)
        else:
            multiplot(cur_axis, w)

        if t:
            cur_axis.set_title(t)
        if xlim:
            cur_axis.set_xlim(xlim[0], xlim[1])
        if ylim:
            cur_axis.set_ylim(ylim[0], ylim[1])
        if hide_ticks:
            cur_axis.set_xticklabels([])
            cur_axis.set_yticklabels([])
            cur_axis.set_xticks([])
            cur_axis.set_yticks([])
        if ax_op:
            cur_axis.axis(ax_op)

        c += 1
        if c >= col:
            r += 1
            c = 0

    fig.savefig(save_name, bbox_inches='tight', figsize=(14,10), dpi=300)
    plt.close()

