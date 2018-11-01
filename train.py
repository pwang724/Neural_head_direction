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
import stationary as st

# set seed for reproducibility
# np.random.seed(2)
# tf.set_random_seed(2)


class RNN:
    """An RNN made to model 1D attractor network"""
    def __init__(self, sess, opts):
        stationary = opts.stationary
        time_steps = opts.time_steps
        state_size = opts.state_size
        velocity_size = opts.velocity_size
        rnn_size = opts.rnn_size

        save_path = opts.save_path
        learning_rate = opts.learning_rate
        time_loss_start = opts.time_loss_start
        time_loss_end= opts.time_loss_end

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.sess = sess
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        if stationary:
            self.x = tf.placeholder(tf.float32, [None, time_steps, state_size], name='input_placeholder')
        else:
            self.x = tf.placeholder(tf.float32, [None, time_steps, state_size + velocity_size],
                                    name='input_placeholder')

        self.y = tf.placeholder(tf.float32, [None, time_steps, state_size], name='output_placeholder')
        inputs_series = tf.unstack(self.x, axis=1)
        labels_series = tf.unstack(self.y, axis=1)

        if stationary:
            self.k, self.weight_dict, trainable_list, self.plot_dict = st.define_stationary_weights(opts)
            rnn_func = st.rnn_stationary
        else:
            self.k, self.weight_dict, trainable_list, self.plot_dict = st.define_nonstationary_weights(opts)
            rnn_func = st.rnn_non_stationary

        init_state = tf.zeros(shape=[self.batch_size, rnn_size], dtype= tf.float32)
        state_series = [init_state]
        for i, current_input in enumerate(inputs_series):
            next_state = rnn_func(self.weight_dict, self.k, state_series[-1], current_input, i, opts)
            state_series.append(next_state)

        self.state_series = state_series
        W_out_tf = self.weight_dict[self.k.W_out]
        self.logits = [tf.matmul(s, W_out_tf) for s in state_series]
        self.predictions = [tf.nn.softmax(l) for l in self.logits]
        losses = [tf.nn.softmax_cross_entropy_with_logits_v2(logits=l, labels=labels)
                  for l, labels in zip(self.logits, labels_series)]

        self.total_loss = tf.reduce_mean(losses[time_loss_start:time_loss_end])

        optimizer= tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.minimize(self.total_loss, var_list= trainable_list)
        self.saver = tf.train.Saver()


    def run_training(self, inputs, labels, opts):
        """
        :param inputs: n x t x d input matrix
        :param labels: n x t x d label matrix
        :return:
        """
        n_epoch = opts.epoch
        save_path = opts.save_path
        file_name = opts.file_name
        stationary = opts.stationary

        train_iter, next_element = utils.make_input(self.x, self.y, opts.batch_size)
        sess.run(train_iter.initializer, feed_dict={self.x: inputs, self.y: labels})
        if stationary:
            st.initialize_stationary_weights(sess, opts, self.weight_dict, self.k)
        else:
            st.initialize_nonstationary_weights(sess, opts, self.weight_dict, self.k)

        loss, logits = [], []
        for ep in range(n_epoch):
            cur_inputs, cur_labels = sess.run(next_element)
            feed_dict = {self.x: cur_inputs, self.y: cur_labels, self.batch_size: opts.batch_size}
            logits, cur_loss, _ = self.sess.run([self.logits, self.total_loss,
                                                 self.train_op], feed_dict=feed_dict)
            if ep % 20 == 0:
                loss.append(cur_loss)
            if (ep+1) % 100 == 0:
                print('[*] Epoch %d  total_loss=%.2f' % (ep, cur_loss))

        path_name = os.path.join("./", save_path, file_name)
        if st: #save matrices to load for non-stationary
            st.save_stationary_weights(self.sess, self.weight_dict, self.k, path_name)
        self.saver.save(self.sess, os.path.join(path_name))

        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].set_title('Loss')
        ax[0].plot(np.squeeze(loss))
        fig.savefig(save_path + '/loss.png', bbox_inches='tight')
        return logits, np.array(loss)

    def run_test(self, inputs, labels, opts):
        """Visualization of trained network."""
        k = self.k
        save_path = opts.save_path
        file_name = opts.file_name
        batch_size = opts.test_batch_size
        checkpoint = os.path.join('./',save_path, file_name)
        self.saver.restore(sess, checkpoint)
        feed_dict = {self.x: inputs, self.y: labels, self.batch_size: batch_size}
        states, predictions, total_loss = \
            self.sess.run([self.logits, self.predictions, self.total_loss], feed_dict=feed_dict)

        #plotting
        cc = 3
        rr = inputs.shape[0]
        c, r = 0,0
        fig, ax = plt.subplots(nrows=rr, ncols=cc)
        for i in range(inputs.shape[0]):
            cur_state = [s[i] for s in states]
            cur_pred = [p[i] for p in predictions]
            cur_label = labels[i,:,:]

            titles = ['Hidden','Predictions','Labels']
            data = [cur_state, cur_pred, cur_label]

            for d, t in zip(data, titles):
                if c == 0:
                    sns.heatmap(d, cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax = ax[r,c], cbar=False)
                else:
                    sns.heatmap(d, cmap='RdBu_r', center=.25, vmin=0, vmax=.5, ax = ax[r,c], cbar=False)
                if r == 0:
                    ax[r,c].set_title(t)
                ax[r,c].axis('off')
                c+=1
                if c >= cc:
                    r += 1
                    c = 0

        plot_dict = {k: self.sess.run(v) for k, v in self.plot_dict.items()}
        W_b = plot_dict[k.W_b]
        plot_dict[k.W_b] = W_b.reshape(1,-1)

        cc, rr = 2, 4
        c, r = 0,0
        fig1, ax = plt.subplots(nrows=rr, ncols=cc)
        for k, w in plot_dict.items():
            sns.heatmap(w, cmap='RdBu_r', vmin=-1, vmax=1, ax= ax[r,c], cbar = False)
            ax[r,c].set_title(k)
            ax[r,c].axis('off')
            ax[r,c].axis('equal')
            c += 1
            if c >= cc:
                r += 1
                c = 0

        fig.savefig(save_path + '/test_trials.png', bbox_inches='tight')
        fig1.savefig(save_path + '/weights.png', bbox_inches='tight')

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stationary', action='store_true', default=False, help='stationary or non-stationary training')

    parser.add_argument('--epoch', type=int, default= int(1e4), help= 'number of epochs')
    parser.add_argument('--batch_size', type=int, default= 5, help= 'batch size')
    parser.add_argument('--test_batch_size', type=int, default= 7, help= 'test batch size')
    parser.add_argument('--n_input', type=int, default= 1000, help= 'number of inputs')
    parser.add_argument('--learning_rate', type=int, default= .002, help= 'learning rate')
    parser.add_argument('--time_steps', type=int, default= 20, help= 'rnn time steps')
    parser.add_argument('--time_loss_start', type=int, default= 1, help= 'start time to assessing loss')
    parser.add_argument('--time_loss_end', type=int, default= 25, help= 'end time for assessing loss')

    parser.add_argument('--load_weights', action='store_true', default= True,
                        help= 'load pre-trained weights on stationary problem?')
    parser.add_argument('--fix_weights', action='store_true', default= False,
                        help= 'hidden weights for state to state trainable?')
    parser.add_argument('--dir_weights', type=str, default='./test/stationary/_.pkl',
                        help='directory of saved weights on stationary problem')

    parser.add_argument('--state_size', type=int, default= 20, help= 'size of state')
    parser.add_argument('--rnn_size', type=int, default= 25, help= 'number of rnn neurons')

    parser.add_argument('--bump_size', type=int, default= 6, help= 'size of bump')
    parser.add_argument('--bump_std', type=int, default= 1.5, help= 'std of bump')

    parser.add_argument('--noise', action='store_true', default=False, help='noise boolean')
    parser.add_argument('--noise_intensity', type=float, default= .25, help= 'noise intensity')
    parser.add_argument('--noise_density', type=float, default= .5, help= 'noise density')

    parser.add_argument('--velocity', action='store_true', default=True, help='velocity boolean')
    parser.add_argument('--velocity_size', type=int, default=2, help='velocity state size')
    parser.add_argument('--velocity_start', type=int, default=5, help='velocity start')
    parser.add_argument('--velocity_gap', type=int, default=3, help='velocity gap')

    parser.add_argument('--save_path', type=str, default='./test/non_stationary', help='save folder')
    parser.add_argument('--file_name', type=str, default='_', help='file name within save path')
    return parser


if __name__ == '__main__':
    parser = arg_parser()
    opts = parser.parse_args()


    with tf.Session() as sess:
        ### Note that the loss can never be zero, as the first input can never produce the right output when the input
        ### weights are the identity matrix.

        rnn = RNN(sess, opts)
        sess.run(tf.global_variables_initializer())
        X, Y, _ = inputs.create_inputs(opts)

        run = True
        if run:
            W_h, loss = rnn.run_training(X, Y, opts)

        test = True
        if test:
            e = opts.test_batch_size
            rnn.run_test(X[:e, :, :], Y[:e, :, :], opts)
            plt.show(block=False)






