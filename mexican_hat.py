import tensorflow as tf
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# set seed for reproducibility
np.random.seed(2)
tf.set_random_seed(2)

def sum_normalize(x):
    """Normalize by sum row-wise."""
    if len(x.shape) > 1:
        x = (x.T / np.sum(x, axis=1)).T
    else:
        # Vector
        x = x / np.sum(x)

    return x

def create_2d_inputs(ix, spread=1, dev=1, create_noise=False, intensity=.5, density=.5):
    """
    Create inputs and labels for training. Training data is in
    :param intensity: proportion of maximum activity noise can reach
    :param density: probability any cell is active
    :return: inputs
    """
    spread = min(spread, STATE_SIZE)
    if (spread % 2) == 0:
        spread += 1

    span = int(np.floor(spread / 2))
    activity = sum_normalize(norm.pdf(np.arange(-span, span + 1), scale=dev))

    labels = np.zeros((len(ix), STATE_SIZE))
    labels[:, :spread] += activity
    labels = np.stack([np.roll(lab, s) for lab, s in zip(labels, ix-span)], axis=0)

    inputs = labels.copy()
    if create_noise:
        # sample noisy positions, sample noise for those positions, add noise to inputs
        assert 0 <= density <= 1, "Density is not between 0 and 1"
        assert 0 <= intensity <= 1, "Intensity is not between 0 and 1"
        max_noise = np.amax(labels[0]) * intensity
        noise = np.random.uniform(low=0, high=max_noise, size=inputs.shape)

        inactive_mask = labels == 0
        noise_sample = np.random.uniform(size=noise.shape)
        noise[noise_sample < density] *= 0  # take density % of noise
        noise *= inactive_mask  # remove noise from true activity

        inputs += noise
        inputs = sum_normalize(inputs)

    return inputs, labels

def prepare_training_data(inputs, labels, timesteps):
    """Pad the inputs with zeros and repeat the labels for timesteps."""
    inputs = inputs[:, np.newaxis, :]
    labels = labels[:, np.newaxis, :]

    input_pad = [inputs] + [np.zeros_like(inputs)] * (timesteps - 1)
    inputs = np.concatenate(input_pad, axis=1)
    labels = np.concatenate([labels] * timesteps, axis=1)
    return inputs, labels

class hatRNN:
    """An RNN made to model hat attractor/Mexican hat network properties."""

    def __init__(self, sess, timesteps, save_name):
        self.x = tf.placeholder(tf.float32, [None, timesteps, STATE_SIZE], name='input_placeholder')
        self.y = tf.placeholder(tf.float32, [None, timesteps, STATE_SIZE], name='output_placeholder')
        self.init_state = tf.placeholder(tf.float32, [None, STATE_SIZE], name='init_state')
        self.sess = sess
        self.timesteps = timesteps
        self.save_name = save_name

        train_dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        train_dataset = train_dataset.shuffle(int(1E6)).batch(tf.cast(batch, tf.int64)).repeat()  # inferred repeat
        self.train_iter = train_dataset.make_initializable_iterator()
        self.next_element = self.train_iter.get_next()

        labels_series = tf.unstack(self.y, axis=1)
        self.W_h = tf.get_variable('W_h', shape=[3, STATE_SIZE, STATE_SIZE])

        states = tf.scan(self.gru_fn, tf.transpose(self.x, [1,0,2]), initializer=self.init_state)
        self.state_series = tf.unstack(states, axis=0)  # output of states leaves t on zero axis
        self.predictions = [tf.nn.softmax(logits) for logits in self.state_series]
        # the logits are the states
        self.losses = [tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
                  for logits, labels in zip(self.state_series, labels_series)]

        self.total_loss = tf.reduce_mean(self.losses)
        self.train_op = tf.train.AdamOptimizer(.001).minimize(self.total_loss)
        self.saver = tf.train.Saver({"W_h": self.W_h})

    def gru_fn(self, hprev, input):
        #  update gate
        z = tf.sigmoid(input + tf.matmul(hprev, self.W_h[0]))
        #  reset gate
        r = tf.sigmoid(input + tf.matmul(hprev, self.W_h[1]))
        #  intermediate
        h = tf.tanh(input + tf.matmul((r * hprev), self.W_h[2]))
        # new state
        st = (1 - z) * h + (z * hprev)
        return st

    def run_training(self, inputs, labels, n_epoch=1e3, batch_size=16, zero_tanh_diag=False, zero_sig_diag=False):
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
            states, loss, t_loss, _ = self.sess.run([self.state_series, self.losses, self.total_loss,
                                                     self.train_op], feed_dict=feed_dict)
            if n % 20 == 0:
                total_loss.append(t_loss)

            if (n+1) % 100 == 0:
                print(n)

        save_path = self.saver.save(self.sess, "./" + self.save_name)
        return states, np.array(total_loss)

    def run_test(self, inputs, labels, checkpoint):
        """Only one example is input for visualization."""
        self.saver.restore(sess, checkpoint)
        init_state = np.zeros((1, STATE_SIZE))
        feed_dict = {self.x: inputs, self.y: labels, self.init_state: init_state}
        states, pred, loss, total_loss = self.sess.run([self.state_series, self.predictions, self.losses,
                                                                       self.total_loss], feed_dict=feed_dict)

        print('test label', labels[0,0,:])
        print('final state', states[-1])
        print('final prediction', np.round(pred[-1], 5))

        f3 = plt.figure()
        plt.subplot(311)
        ax3 = sns.heatmap(np.concatenate(states, axis=0), cmap='RdBu_r', center=0)
        plt.title('States')

        plt.subplot(312)
        ax4 = sns.heatmap(np.concatenate(pred, axis=0), cmap='RdBu_r', center=0)
        plt.title('Predictions')

        plt.subplot(313)
        print(labels[0, 0, :].shape)
        print(states[-1].shape)
        ax5 = sns.heatmap(np.stack([labels[0, 0, :], np.squeeze(pred[-1])]), cmap='RdBu_r', center=0)
        plt.title('Test Label -> Final Prediction')


with tf.Session() as sess:
    STATE_SIZE = 10
    n_epoch = int(1e4)
    batch = 10
    steps = 30

    rnn = hatRNN(sess, steps)
    sess.run(tf.global_variables_initializer())
    ix = np.random.randint(low=0, high=STATE_SIZE, size=10*batch)

    inputs, labels = create_2d_inputs(ix, spread=2, dev=1, create_noise=False, intensity=.5, density=.5)
    inputs, labels = prepare_training_data(inputs, labels, steps)
    inputs = np.concatenate([inputs] * 10, axis=0)
    labels = np.concatenate([labels] * 10, axis=0)
    run = False
    if run:
        weights, loss = rnn.run_training(inputs, labels, n_epoch=n_epoch, batch_size=batch)
        loss_fig = plt.figure()
        plt.title('Losses')
        loss_ax = plt.plot(np.squeeze(loss))

    test = False
    if test:
        load_file = "hatRNN"
        rnn.run_test(inputs[5:6, :, :], labels[5:6, :, :], checkpoint=load_file)

    plt.show()
    ### Note that the loss can never be zero, as the first input can never produce the right output when the input
    ### weights are the identity matrix.


