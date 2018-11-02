import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import config

def make_input(x, y, batch_size):
    data = tf.data.Dataset.from_tensor_slices((x, y))
    data = data.shuffle(int(1E6)).batch(tf.cast(batch_size, tf.int64)).repeat()
    train_iter = data.make_initializable_iterator()
    next_element = train_iter.get_next()
    return train_iter, next_element

def sort_weights(mat, axis, arg_pos= 1):
    if arg_pos:
        temp = np.square(mat * (mat > 0))
    else:
        temp = np.square(mat * (mat < 0))

    if axis == 1:
        ix = np.arange(mat.shape[1]).reshape(-1, 1)
        moment = np.matmul(temp, ix).squeeze() / np.sum(temp, axis=1)
        sort_ix = np.argsort(moment)
        mat_sorted = mat[sort_ix, :]
    else:
        ix = np.arange(mat.shape[0]).reshape(1, -1)
        moment = np.matmul(ix, temp).squeeze() / np.sum(temp, axis=0)
        sort_ix = np.argsort(moment)
        mat_sorted = mat[:, sort_ix]
    return sort_ix, mat_sorted

def pretty_image(tup, col, row, save_name):
    """input: list of tuples
    first arg of each tuple: title
    second arg of each tuple: matrix to plot
    """
    c, r = 0, 0
    fig, ax = plt.subplots(nrows=row, ncols=col)
    for t, w in tup:
        sns.heatmap(w, cmap='RdBu_r', vmin=-1, vmax=1, ax=ax[r, c], cbar=False)
        ax[r, c].set_title(t)
        ax[r, c].axis('off')
        ax[r, c].axis('equal')
        c += 1
        if c >= col:
            r += 1
            c = 0
    fig.savefig(save_name, bbox_inches='tight')

def pretty_plot(tup, col, row, save_name):
    """input: list of tuples
    first arg of each tuple: title
    second arg of each tuple: matrix to plot
    """
    c, r = 0, 0
    fig, ax = plt.subplots(nrows=row, ncols=col)
    for t, w in tup:
        ax[r, c].plot(w)
        ax[r, c].set_title(t)
        c += 1
        if c >= col:
            r += 1
            c = 0
    fig.savefig(save_name, bbox_inches='tight')

def make_modified_path(name):
    n = 0
    modified_path = name + '_' + format(n, '02d')
    while (os.path.exists(modified_path)):
        n += 1
        modified_path = os.path.join(name + '_' + format(n, '02d'))
    os.makedirs(modified_path)
    return modified_path

def save_parameters(path, opts):
    save_name = os.path.join(path, 'parameters.txt')
    super = config.shared_input_config
    super_dict = super.__dict__
    cur_dict = opts.__dict__
    super_dict = {k:v for k, v in super_dict.items() if k[:2] != '__'}
    cur_dict = {k:v for k, v in cur_dict.items() if k[:2] != '__'}

    with open(save_name, 'w') as f:
        for k, v in super_dict.items():
            f.write('%s: %s \n' %(k, v))
        for k, v in cur_dict.items():
            f.write('%s: %s \n' % (k, v))
    pass