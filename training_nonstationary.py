import argparse
import tensorflow as tf
import numpy as np
import inputs
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import scipy.stats as stats
import pickle as pkl
import utils as utils
import rnn_cell as rnn
# set seed for reproducibility
# np.random.seed(2)
# tf.set_random_seed(2)


class RNN:
    """An RNN made to model 1D attractor network"""

    def __init__(self, sess, opts):
        time_steps = opts.time_steps
        state_size = opts.state_size
        velocity_size = opts.velocity_size
        rnn_size = opts.rnn_size

        save_path = opts.save_path
        learning_rate = opts.learning_rate
        time_loss_start = opts.time_loss_start
        time_loss_end= opts.time_loss_end

        fix_weights = opts.fix_weights

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.sess = sess
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.x = tf.placeholder(tf.float32, [None, time_steps, state_size+velocity_size], name='input_placeholder')
        self.y = tf.placeholder(tf.float32, [None, time_steps, state_size], name='output_placeholder')
        inputs_series = tf.unstack(self.x, axis=1)
        labels_series = tf.unstack(self.y, axis=1)

        #input weights
        support_size = rnn_size - state_size
        self.W_i_a = tf.constant(np.eye(state_size), dtype= tf.float32)
        self.W_i_b = tf.get_variable('W_ab', shape= [velocity_size, support_size])

        #output weights
        W_out= np.zeros((rnn_size, state_size))
        np.fill_diagonal(W_out, 1)
        self.W_out = tf.constant(W_out, dtype= tf.float32)

        #hidden weights
        self.W_h_mask = tf.get_variable('W_h_mask', shape=[rnn_size, rnn_size], dtype = tf.float32)
        self.W_h_aa = tf.get_variable('W_h_aa', shape=[state_size, state_size])
        self.W_h_ba = tf.get_variable('W_h_ba', shape=[support_size, state_size])
        self.W_h_ab_bb = tf.get_variable('W_ab_bb', shape=[rnn_size, support_size])
        self.W_b = tf.get_variable('b', shape=[rnn_size], initializer=tf.constant_initializer(0.0))
        # self.W_h = tf.get_variable('W_h', shape=[rnn_size, rnn_size])
        # self.W_b = tf.get_variable('b', shape=[rnn_size], initializer=tf.constant_initializer(0.0))

        init_state = tf.zeros(shape=[self.batch_size, rnn_size], dtype= tf.float32)
        state_series = [init_state]
        for i, current_input in enumerate(inputs_series):
            next_state = rnn.rnn_nonstationary(state_series[-1], current_input, i, state_size)
            state_series.append(next_state)

        self.state_series = state_series
        self.logits = [tf.matmul(s, self.W_out) for s in state_series]
        self.predictions = [tf.nn.softmax(l) for l in self.logits]
        losses = [tf.nn.softmax_cross_entropy_with_logits_v2(logits=l, labels=labels)
                  for l, labels in zip(self.logits, labels_series)]

        self.total_loss = tf.reduce_mean(losses[time_loss_start:time_loss_end])
        optimizer = tf.train.AdamOptimizer(learning_rate)
        if fix_weights:
            self.train_op = optimizer.minimize(
                self.total_loss, var_list=[self.W_i_b, self.W_h_ba, self.W_h_ab_bb])
        else:
            self.train_op = optimizer.minimize(
                self.total_loss, var_list=[self.W_i_b, self.W_h_aa, self.W_h_ba, self.W_h_ab_bb])

        self.saver = tf.train.Saver()



    def initialize_weights(self, opts):
        state_size = opts.state_size
        load_weights = opts.load_weights
        dir_weights = opts.dir_weights

        if load_weights:
            with open(dir_weights, 'rb') as f:
                w_dict = pkl.load(f)
            W_h_old = w_dict['W_h']
            W_b_old = w_dict['W_b']

            W_h_aa = self.W_h_aa.eval()
            W_h_ba = self.W_h_ba.eval()
            W_h_ab_bb = self.W_h_ab_bb.eval()

            bigmat = np.hstack((np.vstack((W_h_aa, W_h_ba)), W_h_ab_bb))
            bigmat[:len(W_h_old), :len(W_h_old)] = W_h_old

            self.sess.run(tf.assign(self.W_h_aa, bigmat[:state_size,:state_size]))
            self.sess.run(tf.assign(self.W_h_ba, bigmat[state_size:,:state_size]))
            self.sess.run(tf.assign(self.W_h_ab_bb, bigmat[:,state_size:]))

            W_b = self.W_b.eval()
            W_b[:len(W_b_old)] = W_b_old
            self.sess.run(tf.assign(self.W_b, W_b))
        else:
            W_h_aa = self.W_h_aa.eval()
            np.fill_diagonal(W_h_aa, 0)
            self.sess.run(tf.assign(self.W_h_aa, W_h_aa))

        W_h_mask = self.W_h_mask.eval()
        W_h_mask[:,:] = 1
        np.fill_diagonal(W_h_mask, 0)
        self.sess.run(tf.assign(self.W_h_mask, W_h_mask))

    def run_training(self, inputs, labels, opts):
        """
        :param inputs: n x t x d input matrix
        :param labels: n x t x d label matrix
        :return:
        """
        n_epoch = opts.epoch
        save_path = opts.save_path
        file_name = opts.file_name

        train_iter, next_element = utils.make_input(self.x, self.y, opts.batch_size)
        sess.run(train_iter.initializer, feed_dict={self.x: inputs, self.y: labels})
        self.initialize_weights(opts)

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

        self.saver.save(self.sess, os.path.join("./",save_path, file_name))

        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].set_title('Loss')
        ax[0].plot(np.squeeze(loss))
        fig.savefig(save_path + '/loss.png', bbox_inches='tight')
        return logits, np.array(loss)

    def run_test(self, inputs, labels, opts):
        """Visualization of trained network."""
        save_path = opts.save_path
        file_name = opts.file_name
        batch_size = opts.test_batch_size
        checkpoint = os.path.join('./',save_path, file_name)
        self.saver.restore(sess, checkpoint)
        feed_dict = {self.x: inputs, self.y: labels, self.batch_size: batch_size}
        states, predictions, total_loss = \
            self.sess.run([self.logits, self.predictions, self.total_loss], feed_dict=feed_dict)
        W_aa, W_ba, W_ab_bb, W_bias, W_i_b, W_o = self.sess.run(
            [self.W_h_aa, self.W_h_ba, self.W_h_ab_bb, self.W_b, self.W_i_b, self.W_out])

        W_h = np.hstack((np.vstack((W_aa, W_ba)), W_ab_bb))
        W_i = W_i_b

        #testing
        W_i_a, W_mask = self.sess.run([self.W_i_a, self.W_h_mask])

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

        state_size = opts.state_size
        W_h_sorted = np.copy(W_h)
        cols = W_h_sorted[:, state_size:]
        ix = np.arange(cols.shape[0]).reshape(1,-1)
        moment = np.matmul(ix, cols).flatten()
        sort_ix = np.argsort(moment)
        W_h_sorted[:,state_size:] = W_h_sorted[:, state_size+sort_ix]
        W_h_sorted[state_size:,:] = W_h_sorted[state_size+sort_ix, :]

        cc = 2
        rr = 2
        c, r = 0,0
        fig1, ax = plt.subplots(nrows=rr, ncols=cc)
        data = [W_h, W_o, W_i, W_bias.reshape(1,-1)]
        title = ['W_h','W_o', 'W_i', 'W_biases']
        for i, w in enumerate(data):
            sns.heatmap(w, cmap='RdBu_r', vmin=-1, vmax=1, ax= ax[r,c], cbar = False)
            ax[r,c].set_title(title[i])
            ax[r,c].axis('off')
            ax[r,c].axis('equal')
            c += 1
            if c >= cc:
                r += 1
                c = 0

        fig.savefig(save_path + '/test.png', bbox_inches='tight')
        fig1.savefig(save_path + '/weights.png', bbox_inches='tight')

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default= int(1e3), help= 'number of epochs')
    parser.add_argument('--batch_size', type=int, default= 5, help= 'batch size')
    parser.add_argument('--test_batch_size', type=int, default= 6, help= 'test batch size')
    parser.add_argument('--n_input', type=int, default= 1000, help= 'number of inputs')
    parser.add_argument('--learning_rate', type=int, default= .001, help= 'learning rate')
    parser.add_argument('--time_steps', type=int, default= 50, help= 'rnn time steps')
    parser.add_argument('--time_loss_start', type=int, default= 0, help= 'start time to assessing loss')
    parser.add_argument('--time_loss_end', type=int, default= 50, help= 'end time for assessing loss')

    parser.add_argument('--load_weights', action='store_true', default= True,
                        help= 'load pre-trained weights on stationary problem?')
    parser.add_argument('--fix_weights', action='store_true', default= False,
                        help= 'hidden weights for state to state trainable?')

    parser.add_argument('--state_size', type=int, default= 20, help= 'size of state')
    parser.add_argument('--rnn_size', type=int, default= 40, help= 'number of rnn neurons')

    parser.add_argument('--bump_size', type=int, default= 6, help= 'size of bump')
    parser.add_argument('--bump_std', type=int, default= 1.5, help= 'std of bump')

    parser.add_argument('--noise', action='store_true', default=False, help='noise boolean')
    parser.add_argument('--noise_intensity', type=float, default= .25, help= 'noise intensity')
    parser.add_argument('--noise_density', type=float, default= .5, help= 'noise density')

    parser.add_argument('--velocity', action='store_true', default=True, help='velocity boolean')
    parser.add_argument('--velocity_size', type=int, default=2, help='velocity state size')
    parser.add_argument('--velocity_start', type=int, default=5, help='velocity start')
    parser.add_argument('--velocity_gap', type=int, default=3, help='velocity gap')

    parser.add_argument('--dir_weights', type=str, default='./stationary/_/_.pkl',
                        help='directory of saved weights on stationary problem')
    parser.add_argument('--save_path', type=str, default='./non_stationary/_', help='save folder')
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

        run = False
        if run:
            weights, loss = rnn.run_training(X, Y, opts)

        test = True
        if test:
            e = opts.test_batch_size
            rnn.run_test(X[:e, :, :], Y[:e, :, :], opts)
            plt.show(block=False)






