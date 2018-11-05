import tensorflow as tf
import numpy as np
import inputs
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle as pkl

import utils as utils
import rnn as rnn_helper
import config

# set seed for reproducibility
# np.random.seed(2)
# tf.set_random_seed(2)

def create_tf_dataset(x, y, batch_size):
    data = tf.data.Dataset.from_tensor_slices((x, y))
    data = data.shuffle(int(1E6)).batch(tf.cast(batch_size, tf.int64)).repeat()
    train_iter = data.make_initializable_iterator()
    next_element = train_iter.get_next()
    next_X, next_Y = next_element[0], next_element[1]
    return train_iter, next_X, next_Y

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

def modify_path(path):
    n = 0
    path_mod = path + '_' + format(n, '02d')
    while (os.path.exists(path_mod)):
        n += 1
        path_mod = os.path.join(path + '_' + format(n, '02d'))
    os.makedirs(path_mod)
    return path_mod

def save_pickle(weight_dict, pathname):
    sess = tf.get_default_session()
    save_dict = {k: sess.run(v) for k, v in weight_dict.items()}
    with open(pathname + ".pkl", 'wb') as f:
        pkl.dump(save_dict, f)

def save_parameters(path, opts):
    if not os.path.exists(path):
        os.makedirs(path)
    save_name = os.path.join(path, 'parameters.txt')
    cur_dict = opts.__dict__
    cur_dict = {k: v for k, v in cur_dict.items() if k[:2] != '__'}
    with open(save_name, 'w') as f:
        for k, v in cur_dict.items():
            f.write('%s: %s \n' % (k, v))

class RNN:
    """An RNN made to model 1D attractor network"""
    def __init__(self, opts):
        self.opts = opts

        X_pl, Y_pl = create_placeholders(opts)
        train_iter, next_X, next_Y = create_tf_dataset(X_pl, Y_pl, opts.batch_size)
        self.model(next_X, next_Y)
        self.train_iter = train_iter
        self.X_pl = X_pl
        self.Y_pl = Y_pl
        self.next_Y = next_Y

    def model(self, x, y):
        opts = self.opts
        stationary = opts.stationary
        state_size = opts.state_size
        rnn_size = opts.rnn_size
        batch_size = opts.batch_size

        learning_rate = opts.learning_rate
        time_loss_start = opts.time_loss_start
        time_loss_end= opts.time_loss_end

        inputs_series = tf.unstack(x, axis=1)
        labels_series = tf.unstack(y, axis=1)
        if stationary:
            self.k, self.weight_dict, trainable_list, self.plot_keys = rnn_helper.define_stationary_weights(opts)
            rnn_func = rnn_helper.rnn_stationary
        else:
            self.k, self.weight_dict, trainable_list, self.plot_keys = rnn_helper.define_nonstationary_weights(opts)
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
        self.activity_loss = activity_constant * tf.reduce_mean(extra_neurons_activity)
        self.total_loss = self.xe_loss + self.weight_loss + self.activity_loss

        optimizer= tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.minimize(self.total_loss, var_list= trainable_list)
        self.saver = tf.train.Saver()


    def train(self, opts):
        """
        :param inputs: n x t x d input matrix
        :param labels: n x t x d label matrix
        :return:
        """
        sess = tf.get_default_session()
        n_epoch = opts.epoch
        save_path = opts.save_path
        image_folder = opts.image_folder
        file_name = opts.file_name
        data_name = opts.data_name
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

        cur_loss, xe_loss, weight_loss, activity_loss = 0, 0, 0, 0
        loss_list, xe_loss_list, activity_loss_list, weight_loss_list, logits = [], [], [], [], []
        for ep in range(n_epoch):
            for b in range(n_batch_per_epoch):
                cur_loss, xe_loss, weight_loss, activity_loss, _ = sess.run(
                    [self.total_loss, self.xe_loss, self.weight_loss, self.activity_loss, self.train_op])

            if (ep % 2 == 0 and ep>0): #save to loss file
                loss_list.append(cur_loss)
                xe_loss_list.append(xe_loss)
                activity_loss_list.append(activity_loss)
                weight_loss_list.append(weight_loss)
            if (ep % 5 == 0 and ep>0): #display in terminal
                print('[*] Epoch %d  total_loss=%.2f xe_loss=%.2f a_loss=%.2f, w_loss=%.2f'
                      % (ep, cur_loss, xe_loss, activity_loss, weight_loss))
            if (ep % 50 == 0) and (ep >0): #save files
                # save parameters, save weights, save some test data, save model ckpt
                epoch_path = modify_path(save_path)
                epoch_tf_name = os.path.join(epoch_path, file_name)
                save_parameters(epoch_path, opts)
                save_pickle(self.weight_dict, epoch_tf_name)
                data = {'states': self.states, 'predictions': self.predictions,
                        'labels': self.next_Y}
                save_pickle(data, os.path.join(epoch_path, data_name))
                self.saver.save(sess, epoch_tf_name)

                image_path = os.path.join(epoch_path, image_folder)
                os.makedirs(image_path)
                loss_name = os.path.join(image_path, 'loss.png')
                loss_title = ['Loss','xe loss','activity loss','weight loss']
                losses = [loss_list, xe_loss_list, activity_loss_list, weight_loss_list]
                utils.pretty_plot(zip(loss_title, losses), col=2, row=2, save_name= loss_name)

        #save latest
        save_parameters(save_path, opts)
        save_pickle(self.weight_dict, tf_name)
        data = {'states': self.states, 'predictions': self.predictions,
                'labels': self.next_Y}
        save_pickle(data, os.path.join(save_path, data_name))
        self.saver.save(sess, tf_name)

        image_path = os.path.join(save_path, image_folder)
        os.makedirs(image_path)
        loss_name = os.path.join(image_path, 'loss.png')
        loss_title = ['Loss', 'xe loss', 'activity loss', 'weight loss']
        losses = [loss_list, xe_loss_list, activity_loss_list, weight_loss_list]
        utils.pretty_plot(zip(loss_title, losses), col=2, row=2, save_name=loss_name)
        # self.test(sess, opts, restore=False, save_path=save_path)

if __name__ == '__main__':
    st_model_opts = config.stationary_model_config()
    non_st_model_opts = config.non_stationary_model_config()
    opts = st_model_opts


    X, Y = inputs.create_inputs(opts)
    rnn = RNN(opts)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(rnn.train_iter.initializer, feed_dict={rnn.X_pl: X, rnn.Y_pl: Y})
        rnn.train(opts)






