import tensorflow as tf
import numpy as np
import inputs
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# set seed for reproducibility
np.random.seed(2)
tf.set_random_seed(2)


class RNN:
    """An RNN made to model 1D attractor network"""

    def __init__(self, sess, time_steps, path_name, batch = 10):
        self.x = tf.placeholder(tf.float32, [None, time_steps, STATE_SIZE], name='input_placeholder')
        self.y = tf.placeholder(tf.float32, [None, time_steps, STATE_SIZE], name='output_placeholder')
        self.init_state = tf.placeholder(tf.float32, [None, STATE_SIZE], name='init_state')
        self.sess = sess
        self.time_steps = time_steps
        self.path_name = path_name
        if not os.path.exists(path_name):
            os.makedirs(path_name)

        train_dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        train_dataset = train_dataset.shuffle(int(1E6)).batch(tf.cast(batch, tf.int64)).repeat()
        self.train_iter = train_dataset.make_initializable_iterator()
        self.next_element = self.train_iter.get_next()

        labels_series = tf.unstack(self.y, axis=1)
        self.W_h = tf.get_variable('W_h', shape=[3, STATE_SIZE, STATE_SIZE])
        self.W_b = tf.get_variable('b', [STATE_SIZE], initializer=tf.constant_initializer(0.0))

        states = tf.scan(self.rnn, tf.transpose(self.x, [1, 0, 2]), initializer=self.init_state)
        self.state_series = tf.unstack(states, axis=0)  # output of states leaves t on zero axis
        self.predictions = [tf.nn.softmax(logits) for logits in self.state_series]
        # the logits are the states
        self.losses = [tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
                  for logits, labels in zip(self.state_series, labels_series)]

        self.total_loss = tf.reduce_mean(self.losses)
        self.train_op = tf.train.RMSPropOptimizer(.001).minimize(self.total_loss)
        self.saver = tf.train.Saver({"W_h": self.W_h})

    def gru(self, hprev, input):
        #  update gate
        z = tf.sigmoid(input + tf.matmul(hprev, self.W_h[0]))
        #  reset gate
        r = tf.sigmoid(input + tf.matmul(hprev, self.W_h[1]))
        #  intermediate
        h = tf.tanh(input + tf.matmul((r * hprev), self.W_h[2]))
        # new state
        st = (1 - z) * h + (z * hprev)
        return st

    def rnn(self, hprev, input):
        out = tf.tanh(input + tf.matmul(hprev, self.W_h[0]))
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
            init_state = np.zeros((cur_inputs.shape[0], STATE_SIZE))
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
        init_state = np.zeros((1, STATE_SIZE))
        feed_dict = {self.x: inputs, self.y: labels, self.init_state: init_state}
        states, pred, loss, total_loss = self.sess.run([self.state_series, self.predictions, self.losses,
                                                                       self.total_loss], feed_dict=feed_dict)
        weights = self.sess.run(self.W_h)

        print('test label', labels[0,0,:])
        print('final state', states[-1])
        print('final prediction', np.round(pred[-1], 5))

        fig, ax = plt.subplots(nrows=2, ncols=3)
        sns.heatmap(np.concatenate(states, axis=0), cmap='RdBu_r', center=0, vmin=0, vmax=1, ax = ax[0,0])
        ax[0,0].set_title('Hidden')
        ax[0,0].axis('off')

        sns.heatmap(np.concatenate(pred, axis=0), cmap='RdBu_r', center=0, vmin=0, vmax=1, ax = ax[0,1])
        ax[0,1].set_title('Predictions')
        ax[0,1].axis('off')

        sns.heatmap(labels[0, :, :], cmap='RdBu_r', center=0, vmin=0, vmax=1, ax = ax[0,2])
        ax[0,2].set_title('Labels')
        ax[0,2].axis('off')

        for i,w in enumerate(weights):
            sns.heatmap(w, cmap='RdBu_r', vmin=-1, vmax=1, ax= ax[1,i])
            ax[1,i].axis('off')
            ax[1,i].set_title('W_' + str(i))



with tf.Session() as sess:
    STATE_SIZE = 10
    n_epoch = int(5e3)
    batch_size = 10
    n_samples = 10 * batch_size
    time_steps = 30
    save_path = 'save_path'

    rnn = RNN(sess, time_steps, save_path, batch_size)
    sess.run(tf.global_variables_initializer())
    X, Y, _ = inputs.create_inputs(n_samples, state_size = STATE_SIZE, time_steps=time_steps, bump_size=2,
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
        rnn.run_test(X[0:1, :, :], Y[0:1, :, :], checkpoint=load_file)

    plt.show()
    ### Note that the loss can never be zero, as the first input can never produce the right output when the input
    ### weights are the identity matrix.


