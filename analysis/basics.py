import numpy as np
import utils
import os
import pickle as pkl
import matplotlib.pyplot as plt
from analysis import receptive_field


##printing losses
# image_path = os.path.join(epoch_path, image_folder)
# os.makedirs(image_path)
# loss_name = os.path.join(image_path, 'loss.png')
# loss_title = ['Loss', 'xe loss', 'activity loss', 'weight loss']
# losses = [loss_list, xe_loss_list, activity_loss_list, weight_loss_list]
# utils.pretty_plot(zip(loss_title, losses), col=2, row=2, save_name=loss_name)

def plot_activity(opts, sort_ix = None):
    stationary = opts.stationary
    state_size = opts.state_size
    save_path = opts.save_path
    image_folder = opts.image_folder
    data_name = opts.activity_name

    sort_ix = plot_sorted_weights(opts)

    im_path = os.path.join(save_path,image_folder)
    if not os.path.exists(im_path):
        os.makedirs(im_path)

    with open(os.path.join(save_path, data_name + '.pkl'), 'rb') as f:
        data_dict = pkl.load(f)

    states, predictions, labels = data_dict['states'], data_dict['predictions'], \
                                  data_dict['Y']

    row = 3
    tup = []
    for i in range(row):
        cur_state = np.array([s[i] for s in states])
        cur_state_core = cur_state[:, :state_size]
        cur_state_extra = cur_state[:, state_size:]
        cur_pred = [p[i] for p in predictions]
        cur_label = labels[i, :, :]
        if i < 1:
            tup.append(('Hidden Core', cur_state_core))
            if sort_ix is not None:
                tup.append(('Hidden Extra', cur_state_extra[:,sort_ix]))
            else:
                tup.append(('Hidden Extra', cur_state_extra))
            tup.append(('Prediction', cur_pred))
            tup.append(('Label', cur_label))
        else:
            tup.append(('', cur_state_core))
            if sort_ix is not None:
                tup.append(('', cur_state_extra[:, sort_ix]))
            else:
                tup.append(('', cur_state_extra))
            tup.append(('', cur_pred))
            tup.append(('', cur_label))
    plot_name = os.path.join(save_path, image_folder, 'activity.png')
    utils.subimage_easy(tup, col=4, row=row, save_name=plot_name)


def plot_weights(opts):
    stationary = opts.stationary
    save_path = opts.save_path
    image_folder = opts.image_folder
    weight_name = opts.weight_name

    im_path = os.path.join(save_path,image_folder)
    if not os.path.exists(im_path):
        os.makedirs(im_path)

    with open(os.path.join(save_path, weight_name + '.pkl'), 'rb') as f:
        weight_dict = pkl.load(f)

    plot_name = os.path.join(save_path, image_folder, 'weights.png')
    utils.subimage_easy(weight_dict.items(), col=2, row=4, save_name=plot_name)

def sort_weights(opts):
    """Visualization of trained network."""
    stationary = opts.stationary
    state_size = opts.state_size
    save_path = opts.save_path
    weight_name = opts.weight_name
    rnn_size = opts.rnn_size
    support_size = rnn_size - state_size

    with open(os.path.join(save_path, weight_name + '.pkl'), 'rb') as f:
        weight_dict = pkl.load(f)

    W_h_ab = weight_dict['model/hidden/W_h_ab:0']
    W_h_ba = weight_dict['model/hidden/W_h_ba:0']

    if stationary == 0:
        W_i_b = weight_dict['model/input/W_i_b:0']
        if W_i_b.shape[0] >2:
            ix0,ix1= 2,3
        else:
            ix0,ix1 = 0,1

        diff = W_i_b[ix0, :] - W_i_b[ix1, :]
        weird_ix = np.all(W_i_b[ix0:ix1 + 1, :] < -.5, axis=0)
        diff[weird_ix] = 10
        sort_ix_1 = np.argsort(diff)

        W_h_ab_sorted = W_h_ab[:, sort_ix_1]
        W_i_b_sorted = W_i_b[:, sort_ix_1]

        diff = W_i_b_sorted[ix0, :] - W_i_b_sorted[ix1, :]
        middle = np.argmin(diff < 0)
        weird = np.sum(weird_ix == False)

        # sort relative to W_ab
        left_sort_ix, _ = sort_by_max_weight(W_h_ab_sorted[:, :middle],
                                             axis=0)
        right_sort_ix, _ = sort_by_max_weight(W_h_ab_sorted[:, middle:weird],
                                              axis=0)
        sort_ix_2 = np.hstack((left_sort_ix, right_sort_ix + middle,
                             range(weird, support_size)))
        sort_ix = sort_ix_1[sort_ix_2.astype(np.int)]
    else:
        sort_ix, _ = sort_by_max_weight(W_h_ba, axis=1)
    return sort_ix

def sort_by_max_weight(mat, axis):
    if axis == 1:
        max_ix = np.argmax(mat, axis=1)
        sort_ix = np.argsort(max_ix)
        mat_sorted = mat[sort_ix, :]
    else:
        max_ix = np.argmax(mat, axis=0)
        sort_ix = np.argsort(max_ix)
        mat_sorted = mat[:, sort_ix]
    return sort_ix, mat_sorted

def plot_sorted_weights(opts):
    """Visualization of trained network."""
    sort_ix = sort_weights(opts)
    plt.style.use('dark_background')
    stationary = opts.stationary
    save_path = opts.save_path
    image_folder = opts.image_folder
    weight_name = opts.weight_name

    im_path = os.path.join(save_path,image_folder)
    if not os.path.exists(im_path):
        os.makedirs(im_path)

    with open(os.path.join(save_path, weight_name + '.pkl'), 'rb') as f:
        weight_dict = pkl.load(f)

    # plot sorted_weights
    W_h_ab = weight_dict['model/hidden/W_h_ab:0']
    W_h_ba = weight_dict['model/hidden/W_h_ba:0']
    W_h_bb = weight_dict['model/hidden/W_h_bb:0']
    W_h_ab_sorted = W_h_ab[:, sort_ix]
    W_h_ba_sorted = W_h_ba[sort_ix, :]
    W_h_bb_sorted = W_h_bb[sort_ix, :]
    W_h_bb_sorted = W_h_bb_sorted[:, sort_ix]
    data = [W_h_ab, W_h_ab_sorted, W_h_ba, W_h_ba_sorted, W_h_bb, W_h_bb_sorted]
    titles = ['W_h_ab', 'sorted', 'W_h_ba', 'sorted', 'W_h_bb','sorted']

    if stationary == 0:
        W_i_b = weight_dict['model/input/W_i_b:0']
        W_i_b_sorted = W_i_b[:, sort_ix]
        data.append(W_i_b)
        data.append(W_i_b_sorted)
        titles.append('W_i_b')
        titles.append('sorted')

    plot_name = os.path.join(save_path, image_folder, 'weights__.png')
    utils.subimage_easy(zip(titles, data), col=2, row=4, save_name=plot_name)

if __name__ == '__main__':
    root = '../experiments/vary_randomize_ab_ba/files/01/'
    d0 = root + 'stationary/'
    d1 = root + 'moving/'

    for d in [d0, d1]:
        if os.path.exists(d0):
            opts = utils.load_parameters(d + 'parameters')
            opts.save_path = d
            # plot_activity(opts)
            plot_weights(opts)
            plot_sorted_weights(opts)

            # if opts.stationary == 0:
            #     points, activity, labels = receptive_field.get_activity(opts)
            #     receptive_field.plot_receptive_field(opts, points, activity,
            #                                          plot_stationary=opts.stationary)
            #     receptive_field.plot_receptive_field(opts, points, activity, plot_stationary=True)

