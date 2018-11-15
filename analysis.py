import numpy as np
import utils
import os
import config
import pickle as pkl
import matplotlib.pyplot as plt

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

    sort_ix = messy_weights(opts)

    im_path = os.path.join(save_path,image_folder)
    if not os.path.exists(im_path):
        os.makedirs(im_path)

    with open(os.path.join(save_path, data_name + '.pkl'), 'rb') as f:
        data_dict = pkl.load(f)

    states, predictions, labels = data_dict['states'], data_dict['predictions'], \
                                  data_dict['Y']

    row = 10
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
    utils.pretty_image(tup, col=4, row=row, save_name=plot_name)


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

    weight_dict.pop('W_b', None)
    plot_name = os.path.join(save_path, image_folder, 'weights.png')
    utils.pretty_image(weight_dict.items(), col=2, row=4, save_name=plot_name)

def sort_weights(opts):
    """Visualization of trained network."""
    stationary = opts.stationary
    state_size = opts.state_size
    save_path = opts.save_path
    weight_name = opts.weight_name

    with open(os.path.join(save_path, weight_name + '.pkl'), 'rb') as f:
        weight_dict = pkl.load(f)

    if stationary:
        W_h = weight_dict['model/hidden/W_h:0']
        W_h_ab = W_h[:state_size, state_size:]
        W_h_ba = W_h[state_size:, :state_size]
    else:
        W_h_ab_bb = weight_dict['model/hidden/W_h_ab_bb:0']
        W_h_ab = W_h_ab_bb[:state_size,:]
        W_h_ba = weight_dict['model/hidden/W_h_ba:0']

    if stationary == 0:
        W_i_b = weight_dict['model/input/W_i_b:0']
        diff = W_i_b[0, :] - W_i_b[1, :]
        sort_ix_1 = np.argsort(diff)
        W_h_ab_sorted = W_h_ab[:, sort_ix_1]
        W_i_b_sorted = W_i_b[:, sort_ix_1]

        thres = 2
        diff = W_i_b_sorted[0, :] - W_i_b_sorted[1, :]
        middle = np.argmax(diff > thres)
        smaller = np.argmin(diff < -thres)

        # sort relative to W_ab
        left_sort_ix, _ = utils.sort_weights(W_h_ab_sorted[:, :smaller],
                                             axis=0, arg_pos=1)
        right_sort_ix, _ = utils.sort_weights(W_h_ab_sorted[:, middle:],
                                              axis=0, arg_pos=1)
        sort_ix_2 = np.hstack((left_sort_ix, right_sort_ix + middle,
                             range(smaller, middle)))
        sort_ix = sort_ix_1[sort_ix_2]
    else:
        sort_ix, _ = utils.sort_weights(W_h_ba, axis=1, arg_pos=1)
    return sort_ix

def messy_weights(opts):
    """Visualization of trained network."""
    stationary = opts.stationary
    state_size = opts.state_size
    save_path = opts.save_path
    image_folder = opts.image_folder
    weight_name = opts.weight_name

    im_path = os.path.join(save_path,image_folder)
    if not os.path.exists(im_path):
        os.makedirs(im_path)

    with open(os.path.join(save_path, weight_name + '.pkl'), 'rb') as f:
        weight_dict = pkl.load(f)

    # plot sorted_weights
    if stationary:
        W_h = weight_dict['model/hidden/W_h:0']
        W_h_ab = W_h[:state_size, state_size:]
        W_h_ba = W_h[state_size:, :state_size]
    else:
        W_h_ab_bb = weight_dict['model/hidden/W_h_ab_bb:0']
        W_h_ab = W_h_ab_bb[:state_size,:]
        W_h_ba = weight_dict['model/hidden/W_h_ba:0']

    if stationary == 0:
        W_i_b = weight_dict['model/input/W_i_b:0']
        diff = W_i_b[0, :] - W_i_b[1, :]
        sort_ix_1 = np.argsort(diff)
        W_h_ab_sorted = W_h_ab[:, sort_ix_1]
        W_h_ba_sorted = W_h_ba[sort_ix_1, :]
        W_i_b_sorted = W_i_b[:, sort_ix_1]

        thres = 1
        diff = W_i_b_sorted[0, :] - W_i_b_sorted[1, :]
        middle = np.argmax(diff > thres)
        smaller = np.argmin(diff < -thres)

        # sort relative to W_ab
        left_sort_ix, _ = utils.sort_weights(W_h_ab_sorted[:, :smaller],
                                             axis=0, arg_pos=1)
        right_sort_ix, _ = utils.sort_weights(W_h_ab_sorted[:, middle:],
                                              axis=0, arg_pos=1)
        sort_ix_2 = np.hstack((left_sort_ix, right_sort_ix + middle,
                             range(smaller, middle)))
        W_h_ab_sorted = W_h_ab_sorted[:, sort_ix_2]
        W_h_ba_sorted = W_h_ba_sorted[sort_ix_2, :]
        W_i_b_sorted = W_i_b_sorted[:, sort_ix_2]
        sort_ix = sort_ix_1[sort_ix_2]

        data = [W_h_ab, W_h_ab_sorted, W_h_ba, W_h_ba_sorted, W_i_b, W_i_b_sorted]
        titles = ['W_h_ab', 'W_h_ab_sorted', 'W_h_ba', 'W_h_ba_sorted', 'W_i_b', 'W_i_b_sorted']
    else:
        W_h_ab_sorted = np.copy(W_h_ab)
        W_h_ba_sorted = np.copy(W_h_ba)
        sort_ix, _ = utils.sort_weights(W_h_ba, axis=1, arg_pos=1)
        W_h_ab_sorted = W_h_ab_sorted[:, sort_ix]
        W_h_ba_sorted = W_h_ba_sorted[sort_ix, :]

        data = [W_h_ab, W_h_ab_sorted, W_h_ba, W_h_ba_sorted]
        titles = ['W_h_ab', 'W_h_ab_sorted', 'W_h_ba', 'W_h_ba_sorted']
    plot_name = os.path.join(save_path, image_folder, 'weights__.png')
    utils.pretty_image(zip(titles, data), col=2, row=3, save_name=plot_name)
    return sort_ix

if __name__ == '__main__':
    d = './lab_meeting/100'
    dirs = [os.path.join(d, o) for o in os.listdir(d)
     if os.path.isdir(os.path.join(d, o))]

    for d in dirs:
        opts = utils.load_parameters(d + '/parameters')
        opts.save_path = d
        plot_activity(opts)
        # plot_weights(opts)
        messy_weights(opts)

