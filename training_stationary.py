import tensorflow as tf
import numpy as np
import inputs
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# set seed for reproducibility
# np.random.seed(2)
# tf.set_random_seed(2)


class RNN:
    """An RNN made to model 1D attractor network"""

    def __init__(self, sess, time_steps, path_name, learning_rate, batch = 10):
        self.x = tf.placeholder(tf.float32, [None, time_steps, state_size], name='input_placeholder')
        self.y = tf.placeholder(tf.float32, [None, time_steps, state_size], name='output_placeholder')
        self.init_state = tf.placeholder(tf.float32, [None, state_size], name='init_state')
        self.sess = sess
        self.time_steps = time_steps
        self.path_name = path_name
        if not os.path.exists(path_name):
            os.makedirs(path_name)

        train_dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        train_dataset = train_dataset.shuffle(int(1E6)).batch(tf.cast(batch, tf.int64)).repeat()
        self.train_iter = train_dataset.make_initializable_iterator()
        self.next_element = self.train_iter.get_next()

        inputs_series = tf.unstack(self.x, axis=1)
        labels_series = tf.unstack(self.y, axis=1)
        self.W_h = tf.get_variable('W_h', shape=[3, state_size, state_size])
        self.W_b = tf.get_variable('b', shape=[3, state_size], initializer=tf.constant_initializer(0.0))

        self.state_series = [self.init_state]
        for i, current_input in enumerate(inputs_series):
            next_state = self.rnn(self.state_series[-1], current_input, i)
            self.state_series.append(next_state)

        # states1 = tf.scan(self.rnn, tf.transpose(self.x, [1, 0, 2]), initializer=self.init_state)
        # self.state_series1 = tf.unstack(states1, axis=0)  # output of states leaves t on zero axis
        self.logits = [s for s in self.state_series]
        self.predictions = [tf.nn.softmax(l) for l in self.logits]
        self.losses = [tf.nn.softmax_cross_entropy_with_logits_v2(logits=l, labels=labels)
                  for l, labels in zip(self.logits, labels_series)]

        self.total_loss = tf.reduce_mean(self.losses)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.total_loss)
        self.saver = tf.train.Saver({"W_h": self.W_h, "W_b":self.W_b})

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
        out = tf.tanh(self.W_b[0] + input + tf.matmul(h_prev, self.W_h[0]),
                               name='time_{}'.format(name))
        return out

    def run_training(self, inputs, labels, n_epoch=1000, zero_tanh_diag=False, zero_sig_diag=False):
        """
        :param inputs: n x t x d input matrix
        :param labels: n x t x d label matrix
        :return:
        """
        sess.run(self.train_iter.initializer, feed_dict={self.x: inputs, self.y: labels})
        total_loss = []
        for n in range(n_epoch):
            cur_inputs, cur_labels = sess.run(self.next_element)
            init_state = np.zeros((cur_inputs.shape[0], state_size))
            feed_dict = {self.x: cur_inputs, self.y: cur_labels, self.init_state: init_state}
            states, t_loss, _ = self.sess.run([self.state_series, self.total_loss,
                                                     self.train_op], feed_dict=feed_dict)
            if n % 20 == 0:
                total_loss.append(t_loss)

            if (n+1) % 100 == 0:
                print(n, t_loss)

        self.saver.save(self.sess, "./" + self.path_name + "/stationary")
        return states, np.array(total_loss)

    def run_test(self, inputs, labels, checkpoint):
        """Only one example is input for visualization."""
        self.saver.restore(sess, checkpoint)
        init_state = np.zeros((inputs.shape[0], state_size))
        feed_dict = {self.x: inputs, self.y: labels, self.init_state: init_state}
        states, pred, loss, total_loss = self.sess.run([self.state_series, self.predictions, self.losses,
                                                                       self.total_loss], feed_dict=feed_dict)
        weights, biases = self.sess.run([self.W_h, self.W_b])

        cc = 3
        rr = inputs.shape[0]
        c, r = 0,0
        fig, ax = plt.subplots(nrows=rr, ncols=cc)
        for i in range(inputs.shape[0]):
            cur_state = [s[i] for s in states]
            cur_pred = [p[i] for p in pred]
            cur_label = labels[i,:,:]

            titles = ['Hidden','Predictions','Labels']
            data = [cur_state, cur_pred, cur_label]

            for d, t in zip(data, titles):
                sns.heatmap(d, cmap='RdBu_r', center=0, vmin=0, vmax=1, ax = ax[r,c], cbar=False)
                ax[r,c].set_title(t)
                ax[r,c].axis('off')
                c+=1
                if c >= cc:
                    r += 1
                    c = 0

        fig1, ax = plt.subplots(nrows=2, ncols=3)
        for i,w in enumerate(weights):
            sns.heatmap(w, cmap='RdBu_r', vmin=-1, vmax=1, ax= ax[0,i], cbar = False)
            ax[0,i].axis('off')
            ax[0,i].set_title('W_' + str(i))
            ax[0,i].axis('equal')

        for i,b in enumerate(biases):
            sns.heatmap(b.reshape(1,-1), cmap='RdBu_r', vmin=-1, vmax=1, ax= ax[1,i], cbar = False)
            ax[1,i].axis('off')
            ax[1,i].set_title('B_' + str(i))
            ax[1, i].axis('image')

        path = self.path_name
        fig.savefig(path + '/img0.png', bbox_inches='tight')
        fig1.savefig(path + '/img1.png', bbox_inches='tight')



with tf.Session() as sess:
    state_size = 20
    bump_size = 7
    bump_std = 1.5
    n_epoch = int(1e4)
    batch_size = 5
    n_samples = 1000
    time_steps = 20
    learning_rate = .003
    save_path = 'save_path'

    rnn = RNN(sess, time_steps, save_path, learning_rate, batch_size)
    sess.run(tf.global_variables_initializer())
    X, Y, _ = inputs.create_inputs(n_samples, state_size = state_size, time_steps=time_steps,
                                   bump_size=bump_size, bump_std = bump_std,
                                   noise= False,
                                   velocity=False, velocity_start=4, velocity_gap = 5)

    run = True
    if run:
        weights, loss = rnn.run_training(X, Y, n_epoch=n_epoch)
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].set_title('Loss')
        ax[0].plot(np.squeeze(loss))

    test = True
    if test:
        load_file = save_path + '/stationary'
        rnn.run_test(X[0:7, :, :], Y[0:7, :, :], checkpoint=load_file)

    plt.show(block=False)
    time.sleep(3)
    ### Note that the loss can never be zero, as the first input can never produce the right output when the input
    ### weights are the identity matrix.


