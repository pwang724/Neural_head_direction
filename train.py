import argparse
import tensorflow as tf
import numpy as np
import inputs
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import pickle as pkl

import utils as utils
import rnn as rnn_helper
import config
import shutil

# set seed for reproducibility
# np.random.seed(2)
# tf.set_random_seed(2)

def create_tf_dataset(x, y, batch_size):
    data = tf.data.Dataset.from_tensor_slices((x, y))
    data = data.shuffle(int(1E6)).batch(tf.cast(batch_size, tf.int64)).repeat()
    train_iter = data.make_initializable_iterator()
    next_element = train_iter.get_next()
    return train_iter, next_element

def create_placeholders(opts):
    stationary = opts.stationary
    time_steps = opts.time_steps
    state_size = opts.state_size
    if stationary:
        x = tf.placeholder(tf.float32, [None, time_steps, state_size], name='input_placeholder')
    else:
        velocity_size = opts.velocity_max * 2
        x = tf.placeholder(tf.float32, [None, time_steps, state_size + velocity_size],
                           name='input_placeholder')
    y = tf.placeholder(tf.float32, [None, time_steps, state_size], name='output_placeholder')
    return x, y

class RNN:
    """An RNN made to model 1D attractor network"""
    def __init__(self, x, y, opts):
        stationary = opts.stationary
        state_size = opts.state_size
        rnn_size = opts.rnn_size
        batch_size = opts.batch_size

        learning_rate = opts.learning_rate
        time_loss_start = opts.time_loss_start
        time_loss_end= opts.time_loss_end

        inputs_series = tf.unstack(x, axis=1)
        labels_series = tf.unstack(y, axis=1)
        self.x = x
        self.y = y
        if stationary:
            self.k, self.weight_dict, trainable_list, self.plot_dict = rnn_helper.define_stationary_weights(opts)
            rnn_func = rnn_helper.rnn_stationary
        else:
            self.k, self.weight_dict, trainable_list, self.plot_dict = rnn_helper.define_nonstationary_weights(opts)
            rnn_func = rnn_helper.rnn_non_stationary

        init_state = tf.zeros(shape=[batch_size, rnn_size], dtype= tf.float32)
        state_series = [init_state]
        for i, current_input in enumerate(inputs_series):
            next_state = rnn_func(self.weight_dict, self.k, state_series[-1], current_input, i, opts)
            state_series.append(next_state)

        self.states = state_series
        W_out_tf = self.weight_dict[self.k.W_out]
        self.logits = [tf.matmul(s, W_out_tf) for s in state_series]
        self.predictions = [tf.nn.softmax(l) for l in self.logits]
        xe_losses = [tf.nn.softmax_cross_entropy_with_logits_v2(logits=l, labels=labels)
                  for l, labels in zip(self.logits, labels_series)]
        self.xe_loss = tf.reduce_mean(xe_losses[time_loss_start:time_loss_end])

        W_h = self.weight_dict[self.k.W_h]
        W_h_ab = W_h[:, state_size:]
        W_h_ba = W_h[state_size:,:]
        weight_constant = 0
        self.weight_loss = weight_constant * (tf.reduce_mean(tf.square(W_h_ab)) + tf.reduce_mean(tf.square(W_h_ba)))

        rnn_activity = tf.stack(state_series, axis=1)
        extra_neurons_activity = rnn_activity[:,:,state_size:]
        activity_constant = .1
        self.activity_loss = activity_constant * tf.reduce_mean(tf.square(extra_neurons_activity))
        self.total_loss = self.xe_loss + self.weight_loss + self.activity_loss

        optimizer= tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.minimize(self.total_loss, var_list= trainable_list)
        self.saver = tf.train.Saver()


    def train(self, sess, opts):
        """
        :param inputs: n x t x d input matrix
        :param labels: n x t x d label matrix
        :return:
        """
        n_epoch = opts.epoch
        save_path = opts.save_path
        file_name = opts.file_name
        stationary = opts.stationary
        n_batch_per_epoch = opts.n_input // opts.batch_size

        tf_name = os.path.join(save_path, file_name)
        if opts.load_checkpoint:
            self.saver.restore(sess, tf_name)
        else:
            if stationary:
                rnn_helper.initialize_stationary_weights(sess, opts, self.weight_dict, self.k)
            else:
                rnn_helper.initialize_nonstationary_weights(sess, opts, self.weight_dict, self.k)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        cur_loss, xe_loss, weight_loss, activity_loss = 0, 0, 0, 0
        loss_list, xe_loss_list, activity_loss_list, weight_loss_list, logits = [], [], [], [], []
        for ep in range(n_epoch):
            for b in range(n_batch_per_epoch):
                logits, cur_loss, xe_loss, weight_loss, activity_loss, _ = sess.run(
                    [self.logits, self.total_loss,
                     self.xe_loss, self.weight_loss, self.activity_loss, self.train_op])

            if (ep % 5 == 0 and ep>0):
                loss_list.append(cur_loss)
                xe_loss_list.append(xe_loss)
                activity_loss_list.append(activity_loss)
                weight_loss_list.append(weight_loss)
                print('[*] Epoch %d  total_loss=%.2f xe_loss=%.2f a_loss=%.2f, w_loss=%.2f'
                      % (ep, cur_loss, xe_loss, activity_loss, weight_loss))
            if (ep % 50 == 0) and (ep >0):
                ep_path = utils.make_modified_path(save_path)
                epoch_tf_name = os.path.join(ep_path, file_name)

                utils.save_parameters(ep_path, opts)
                rnn_helper.save_weights(sess, self.weight_dict, epoch_tf_name)
                self.saver.save(sess, epoch_tf_name)

                loss_name = os.path.join(ep_path, 'loss.png')
                loss_title = ['Loss','xe loss','activity loss','weight loss']
                losses = [loss_list, xe_loss_list, activity_loss_list, weight_loss_list]
                utils.pretty_plot(zip(loss_title, losses), col=2, row=2, save_name= loss_name)
                self.test(sess, opts, restore= False, save_path= ep_path)

        #save latest
        utils.save_parameters(save_path, opts)
        rnn_helper.save_weights(sess, self.weight_dict, tf_name)
        self.saver.save(sess, tf_name)
        loss_name = os.path.join(save_path, 'loss.png')
        loss_title = ['Loss', 'xe loss', 'activity loss', 'weight loss']
        losses = [loss_list, xe_loss_list, activity_loss_list, weight_loss_list]
        utils.pretty_plot(zip(loss_title, losses), col=2, row=2, save_name=loss_name)
        self.test(sess, opts, restore=False, save_path=save_path)

    def test(self, sess, opts, restore = True, save_path = None):
        """Visualization of trained network."""
        k = self.k
        stationary = opts.stationary
        state_size = opts.state_size

        if save_path is None:
            save_path = opts.save_path
        if restore:
                file_name = opts.file_name
                checkpoint = os.path.join('./', save_path, file_name)
                self.saver.restore(sess, checkpoint)

        states, predictions, total_loss, labels = \
            sess.run([self.states, self.predictions, self.total_loss, self.y])
        plot_dict = {k: sess.run(v) for k, v in self.plot_dict.items()}

        #plot sorted_weights
        W_h = plot_dict[k.W_h]
        W_h_ab = W_h[:state_size, state_size:]
        W_h_ba = W_h[state_size:, :state_size]

        if stationary == 2:
            # apply left/right sort
            W_i_b = plot_dict[k.W_i_b]
            diff = W_i_b[0,:] - W_i_b[1,:]
            sort_ix = np.argsort(diff)
            W_h_ab_sorted = W_h_ab[:,sort_ix]
            W_h_ba_sorted = W_h_ba[sort_ix,:]
            W_i_b_sorted = W_i_b[:,sort_ix]

            #apply sort to each one
            diff = W_i_b_sorted[0,:]-W_i_b_sorted[1,:]
            split_ix = np.argmax(diff>0)
            left_sort_ix, _ = utils.sort_weights(W_h_ba_sorted[:split_ix,:], axis= 1, arg_pos= 1)
            right_sort_ix, _ = utils.sort_weights(W_h_ba_sorted[split_ix:,:], axis= 1, arg_pos= 1)
            # left_sort_ix, _ = utils.sort_weights(W_h_ab_sorted[:,:split_ix], axis= 0, arg_pos= 1)
            # right_sort_ix, _ = utils.sort_weights(W_h_ab_sorted[:,split_ix:], axis= 0, arg_pos= 1)
            sort_ix = np.hstack((left_sort_ix, right_sort_ix+split_ix))
            W_h_ab_sorted = W_h_ab_sorted[:,sort_ix]
            W_h_ba_sorted = W_h_ba_sorted[sort_ix,:]
            W_i_b_sorted = W_i_b_sorted[:,sort_ix]

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
        plot_name = save_path + '/sorted_weights.png'
        utils.pretty_image(zip(titles, data), col=2, row=3, save_name=plot_name)

        #plot activity
        rr = opts.batch_size
        tup = []
        for i in range(rr):
            cur_state = np.array([s[i] for s in states])
            cur_state_core = cur_state[:,:state_size]
            cur_state_extra = cur_state[:,state_size:]
            cur_pred = [p[i] for p in predictions]
            cur_label = labels[i,:,:]
            if i < 1:
                tup.append(('Hidden Core', cur_state_core))
                tup.append(('Hidden Extra', cur_state_extra[:,sort_ix]))
                tup.append(('Prediction', cur_pred))
                tup.append(('Label', cur_label))
            else:
                tup.append(('', cur_state_core))
                tup.append(('', cur_state_extra[:,sort_ix]))
                tup.append(('', cur_pred))
                tup.append(('', cur_label))
        plot_name = save_path + '/test_trials.png'
        utils.pretty_image(tup, col=4, row=rr, save_name=plot_name)

        #plot weights
        W_b = plot_dict[k.W_b]
        plot_dict[k.W_b] = W_b.reshape(1,-1)
        plot_name = save_path + '/weights.png'
        utils.pretty_image(plot_dict.items(), col= 2, row=3, save_name= plot_name)

if __name__ == '__main__':
    st_model_opts = config.stationary_model_config()
    non_st_model_opts = config.non_stationary_model_config()
    opts = st_model_opts


    X, Y = inputs.create_inputs(opts)
    X_pl, Y_pl = create_placeholders(opts)
    train_iter, next_element = create_tf_dataset(X_pl, Y_pl, opts.batch_size)
    rnn = RNN(next_element[0], next_element[1], opts)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_iter.initializer, feed_dict={X_pl: X, Y_pl: Y})
        rnn.train(sess, opts)






