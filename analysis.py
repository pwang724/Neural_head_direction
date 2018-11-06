import numpy as np
import utils
import os
import config
import pickle as pkl

def plot_activity(opts, sort_ix = None):
    stationary = opts.stationary
    state_size = opts.state_size
    save_path = opts.save_path
    image_folder = opts.image_folder
    data_name = opts.data_name

    with open(os.path.join(save_path, data_name + '.pkl'), 'rb') as f:
        data_dict = pkl.load(f)

    states, predictions, labels = data_dict['states'], data_dict['predictions'], \
                                  data_dict['labels']

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
            # if sort_ix is not None:
            #     tup.append(('Hidden Extra', cur_state_extra[:, sort_ix]))
            # else:
            tup.append(('Hidden Extra', cur_state_extra))
            tup.append(('Prediction', cur_pred))
            tup.append(('Label', cur_label))
        else:
            tup.append(('', cur_state_core))
            # if sort_ix is not None:
            #     tup.append(('', cur_state_extra[:, sort_ix]))
            # else:
            tup.append(('', cur_state_extra))
            tup.append(('', cur_pred))
            tup.append(('', cur_label))
    plot_name = os.path.join(save_path, image_folder, 'activity.png')
    utils.pretty_image(tup, col=4, row=row, save_name=plot_name)

def plot_weights(opts):
    stationary = opts.stationary
    save_path = opts.save_path
    image_folder = opts.image_folder
    weight_name = opts.file_name
    with open(os.path.join(save_path, weight_name + '.pkl'), 'rb') as f:
        weight_dict = pkl.load(f)

    weight_dict.pop('W_b', None)
    weight_dict.pop('W_h_z',None)
    weight_dict.pop('W_h_r',None)
    plot_name = os.path.join(save_path, image_folder, 'weights.png')
    utils.pretty_image(weight_dict.items(), col=2, row=4, save_name=plot_name)


def analyze_weights(opts):
    """Visualization of trained network."""
    stationary = opts.stationary
    state_size = opts.state_size
    save_path = opts.save_path
    image_folder = opts.image_folder
    weight_name = opts.file_name

    with open(os.path.join(save_path, weight_name + '.pkl'), 'rb') as f:
        weight_dict = pkl.load(f)

    # plot sorted_weights
    W_h = weight_dict['W_h']
    W_h_ab = W_h[:state_size, state_size:]
    W_h_ba = W_h[state_size:, :state_size]

    if stationary == 0:
        # apply left/right sort
        W_i_b = weight_dict['W_i_b']
        diff = W_i_b[0, :] - W_i_b[1, :]
        sort_ix = np.argsort(diff)
        W_h_ab_sorted = W_h_ab[:, sort_ix]
        W_h_ba_sorted = W_h_ba[sort_ix, :]
        W_i_b_sorted = W_i_b[:, sort_ix]

        # apply sort to each one
        diff = W_i_b_sorted[0, :] - W_i_b_sorted[1, :]
        split_ix = np.argmax(diff > 0)
        left_sort_ix, _ = utils.sort_weights(W_h_ba_sorted[:split_ix, :], axis=1, arg_pos=1)
        right_sort_ix, _ = utils.sort_weights(W_h_ba_sorted[split_ix:, :], axis=1, arg_pos=1)
        # left_sort_ix, _ = utils.sort_weights(W_h_ab_sorted[:,:split_ix], axis= 0, arg_pos= 1)
        # right_sort_ix, _ = utils.sort_weights(W_h_ab_sorted[:,split_ix:], axis= 0, arg_pos= 1)
        sort_ix = np.hstack((left_sort_ix, right_sort_ix + split_ix))
        W_h_ab_sorted = W_h_ab_sorted[:, sort_ix]
        W_h_ba_sorted = W_h_ba_sorted[sort_ix, :]
        W_i_b_sorted = W_i_b_sorted[:, sort_ix]

        # sort_ix, W_h_ba_sorted = utils.sort_weights(W_h_ba, axis=1, arg_pos=1)
        # full_sort_ix = sort_ix + state_size
        # W_h_sorted = np.copy(W_h)
        # W_h_sorted[state_size:, :] = W_h_sorted[full_sort_ix, :]
        # W_h_sorted[:, state_size:] = W_h_sorted[:, full_sort_ix]

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

if __name__ == '__main__':
    st_model_opts = config.stationary_model_config()
    non_st_model_opts = config.non_stationary_model_config()
    opts = st_model_opts

    d = './curriculum'
    dirs = [os.path.join(d, o) for o in os.listdir(d)
     if os.path.isdir(os.path.join(d, o))]

    for d in dirs:
        opts.save_path = d
        plot_activity(opts)
        plot_weights(opts)
        analyze_weights(opts)

