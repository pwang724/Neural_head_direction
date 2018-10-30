import argparse
import tensorflow as tf
import numpy as np
import inputs
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import scipy.stats as stats

# set seed for reproducibility
# np.random.seed(2)
# tf.set_random_seed(2)


class RNN:
    """An RNN made to model 1D attractor network"""

    def __init__(self, sess, opts):
        time_steps = opts.time_steps
        state_size = opts.state_size
        save_path = opts.save_path
        learning_rate = opts.learning_rate
        rnn_size = opts.rnn_size
        time_loss_start = opts.time_loss_start
        time_loss_end= opts.time_loss_end

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.sess = sess
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.x = tf.placeholder(tf.float32, [None, time_steps, state_size], name='input_placeholder')
        self.y = tf.placeholder(tf.float32, [None, time_steps, state_size], name='output_placeholder')

        inputs_series = tf.unstack(self.x, axis=1)
        labels_series = tf.unstack(self.y, axis=1)

        W_in = np.zeros((state_size, rnn_size))
        np.fill_diagonal(W_in, 1)
        self.W_in = tf.constant(W_in, dtype= tf.float32)

        W_out= np.zeros((rnn_size, state_size))
        np.fill_diagonal(W_out, 1)
        self.W_out = tf.constant(W_out, dtype= tf.float32)

        self.W_h_mask = tf.get_variable('W_h_mask', shape=[rnn_size, rnn_size], dtype = tf.float32, trainable= False)
        self.W_h = tf.get_variable('W_h', shape=[rnn_size, rnn_size])
        self.W_b = tf.get_variable('b', shape=[rnn_size], initializer=tf.constant_initializer(0.0))

        init_state = tf.zeros(shape=[self.batch_size, rnn_size], dtype= tf.float32)
        state_series = [init_state]
        for i, current_input in enumerate(inputs_series):
            next_state = self.rnn(state_series[-1], current_input, i)
            state_series.append(next_state)

        self.state_series = state_series
        self.logits = [tf.matmul(s, self.W_out) for s in state_series]
        self.predictions = [tf.nn.softmax(l) for l in self.logits]
        losses = [tf.nn.softmax_cross_entropy_with_logits_v2(logits=l, labels=labels)
                  for l, labels in zip(self.logits, labels_series)]

        self.total_loss = tf.reduce_mean(losses[time_loss_start:time_loss_end])
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.total_loss, var_list=[self.W_h, self.W_b])
        self.saver = tf.train.Saver({"W_h": self.W_h, "W_b":self.W_b, "W_h_mask": self.W_h_mask})

    def make_input(self, batch_size):
        data = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        data = data.shuffle(int(1E6)).batch(tf.cast(batch_size, tf.int64)).repeat()
        train_iter = data.make_initializable_iterator()
        next_element = train_iter.get_next()
        return train_iter, next_element

    def initialize_weights(self, opt):
        state_size = opt.state_size
        rnn_size = opt.rnn_size

        W_h_mask = self.W_h_mask.eval()
        W_h_mask[:,:] = 1
        # np.fill_diagonal(W_h_mask[:state_size, :state_size], 0)
        np.fill_diagonal(W_h_mask, 0)

        W_h = self.W_h.eval()
        # np.fill_diagonal(W_h[:state_size, :state_size], 0)
        np.fill_diagonal(W_h, 0)

        self.sess.run(tf.assign(self.W_h_mask, W_h_mask))
        self.sess.run(tf.assign(self.W_h, W_h))

    def gru(self, h_prev, input, name):
        #  update gate
        z = tf.sigmoid(input + tf.matmul(h_prev, self.W_h[0]))
        #  reset gate
        r = tf.sigmoid(input + tf.matmul(h_prev, self.W_h[1]))
        #  intermediate
        h = tf.tanh(input + tf.matmul((r * h_prev), self.W_h[2]), name='time_{}'.format(name))
        # new state
        st = (1 - z) * h + (z * h_prev)
        return st

    def rnn(self, h_prev, input, name):
        W_h = tf.multiply(self.W_h, self.W_h_mask)
        out = tf.tanh(self.W_b + tf.matmul(input,self.W_in) + tf.matmul(h_prev, W_h),
                      name='time_{}'.format(name))
        return out

    def run_training(self, inputs, labels, opts):
        """
        :param inputs: n x t x d input matrix
        :param labels: n x t x d label matrix
        :return:
        """
        n_epoch = opts.epoch
        save_path = opts.save_path
        file_name = opts.file_name

        train_iter, next_element = self.make_input(opts.batch_size)
        sess.run(train_iter.initializer, feed_dict={self.x: inputs, self.y: labels})
        self.initialize_weights(opts)
        plt.imshow(sess.run(self.W_h_mask))

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
        weights, biases = self.sess.run([self.W_h, self.W_b])

        #testing
        w_in, w_out, mask = self.sess.run([self.W_in, self.W_out, self.W_h_mask])
        print(mask)

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

        fig1, ax = plt.subplots(nrows=2, ncols=2)
        state_size = opts.state_size
        weights_ = np.copy(weights)
        cols = weights_[:, state_size:]
        ix = np.arange(cols.shape[0]).reshape(1,-1)
        moment = np.matmul(ix, cols).flatten()
        sort_ix = np.argsort(moment)
        weights_[:,state_size:] = weights_[:, state_size+sort_ix]
        weights_[state_size:,:] = weights_[state_size+sort_ix, :]

        for i, w in enumerate([weights, weights_]):
            sns.heatmap(w, cmap='RdBu_r', vmin=-1, vmax=1, ax= ax[0,i], cbar = False)
            ax[0,i].set_title('W')
            ax[0,i].axis('off')
            ax[0,i].axis('equal')

        # for i,b in enumerate(biases):
        sns.heatmap(biases.reshape(1,-1), cmap='RdBu_r', vmin=-1, vmax=1, ax= ax[1,0], cbar = False)
        ax[1,0].axis('off')
        ax[1,0].set_title('B')
        ax[1,0].axis('image')

        fig.savefig(save_path + '/img0.png', bbox_inches='tight')
        fig1.savefig(save_path + '/img1.png', bbox_inches='tight')

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default= int(2e4), help= 'number of epochs')
    parser.add_argument('--batch_size', type=int, default= 5, help= 'batch size')
    parser.add_argument('--test_batch_size', type=int, default= 7, help= 'test batch size')
    parser.add_argument('--n_input', type=int, default= 1000, help= 'number of inputs')
    parser.add_argument('--learning_rate', type=int, default= .001, help= 'learning rate')
    parser.add_argument('--time_steps', type=int, default= 20, help= 'rnn time steps')
    parser.add_argument('--time_loss_start', type=int, default= 5, help= 'start time to assessing loss')
    parser.add_argument('--time_loss_end', type=int, default= 15, help= 'end time for assessing loss')

    parser.add_argument('--state_size', type=int, default= 20, help= 'size of state')
    parser.add_argument('--rnn_size', type=int, default= 40, help= 'number of rnn neurons')

    parser.add_argument('--bump_size', type=int, default= 7, help= 'size of bump')
    parser.add_argument('--bump_std', type=int, default= 1, help= 'std of bump')

    parser.add_argument('--noise', action='store_true', default=False, help='noise boolean')
    parser.add_argument('--noise_intensity', type=float, default= .25, help= 'noise intensity')
    parser.add_argument('--noise_density', type=float, default= .5, help= 'noise density')

    parser.add_argument('--velocity', action='store_true', default=False, help='velocity boolean')
    parser.add_argument('--velocity_start', type=int, default=5, help='velocity start')
    parser.add_argument('--velocity_gap', type=int, default=5, help='velocity gap')

    parser.add_argument('--save_path', type=str, default='./test/_', help='save folder')
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
            weights, loss = rnn.run_training(X, Y, opts)

        test = True
        if test:
            e = opts.test_batch_size
            rnn.run_test(X[:e, :, :], Y[:e, :, :], opts)
            plt.show(block=False)






