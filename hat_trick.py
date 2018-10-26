import tensorflow as tf
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

STATE_SIZE = 10

np.random.seed(2)
tf.set_random_seed(2)

n_epoch = int(4e3)
batch = 10

def softmax(x):
    """Compute the softmax function for each row of the input x.

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape
    if x.dtype == int:
        x = x.astype(float)

    if len(x.shape) > 1:
        # vectorized softmax
        x = (x.T - np.amax(x, axis=0))  # now column-wise
        x = np.exp(x)
        x = (x / np.sum(x, axis=0)).T  # back to row-wise

    else:
        # Vector
        x = np.exp(x - np.amax(x))  # normalization to max of 0
        x /= np.sum(x)

    assert x.shape == orig_shape
    return x

def sum_normalize(x):
    """Normalize by sum row-wise."""
    if len(x.shape) > 1:
        x = (x.T / np.sum(x, axis=1)).T

    else:
        # Vector
        x = x / np.sum(x)

    return x

def create_2d_inputs(ix, spread=1, dev=1, create_noise=False, intensity=.5, density=.5, equal_ex=False):
    """
    :param labels: output of create_labels
    :param intensity: proportion of maximum activity noise can reach
    :param density: probability any cell is active
    :return: inputs
    """
    spread = min(spread, STATE_SIZE)
    if (spread % 2) == 0:
        spread += 1

    span = int(np.floor(spread / 2))
    activity = sum_normalize(norm.pdf(np.arange(-span, span + 1), scale=dev))

    if equal_ex:
        ix = np.arange(STATE_SIZE)

    labels = np.zeros((len(ix), STATE_SIZE))
    labels[:, :spread] += activity
    labels = np.stack([np.roll(lab, s) for lab, s in zip(labels, ix-span)], axis=0)

    inputs = labels.copy()
    if create_noise:
        # sample noisy positions, sample noise for those positions, add noise to inputs
        assert 0 <= density <= 1, "Density is not between 0 and 1"
        assert 0 <= intensity <= 1, "Intensity is not between 0 and 1"
        max_noise = np.amax(labels[0]) * intensity
        # noise = np.abs(np.random.normal(scale=dev, size=(len(inputs), STATE_SIZE)))
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

def label_test():
    ix = np.array([9,8,4])
    inputs, labels = create_2d_inputs(ix, spread=3, dev=1, create_noise=True, intensity=.5, density=.5)
    # print(in1)
    inputs, labels = prepare_training_data(inputs, labels, 5)
    print(inputs)
    print(labels)


class hatRNN:
    """An RNN made to model hat attractor/Mexican hat network properties."""

    def __init__(self, sess, timesteps=5):
        # self.x = tf.get_variable("inputs", shape=[1, timesteps, STATE_SIZE], initializer=tf.initializers.zeros)
        # self.y = tf.get_variable("outputs", shape=[1, timesteps, STATE_SIZE], initializer=tf.initializers.zeros)
        # self.init_state = tf.get_variable("init_state", shape=[])

        self.x = tf.placeholder(tf.float32, [None, timesteps, STATE_SIZE], name='input_placeholder')  # using scan, can set timesteps to None
        self.y = tf.placeholder(tf.float32, [None, timesteps, STATE_SIZE], name='output_placeholder')
        self.init_state = tf.placeholder(tf.float32, [None, STATE_SIZE], name='init_state')
        self.sess = sess
        self.timesteps = timesteps
        self.the_mid_be_hittin_tho = tf.constant(1-np.eye(STATE_SIZE), name='the_mid_be_hittin_tho', dtype=tf.float32)

        train_dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        train_dataset = train_dataset.shuffle(int(1E6)).batch(tf.cast(batch, tf.int64))  #.repeat()  # inferred repeat
        self.train_iter = train_dataset.make_initializable_iterator()
        self.next_element = self.train_iter.get_next()

        inputs_series = tf.unstack(self.x, axis=1)
        labels_series = tf.unstack(self.y, axis=1)

        self.W = tf.get_variable('hidden_weights', shape=[STATE_SIZE, STATE_SIZE])

        self.state_series = [self.init_state]
        for i, current_input in enumerate(inputs_series):
            next_state = tf.nn.relu(tf.matmul(self.state_series[i], self.W) + current_input,
                                 name='Activity_{}'.format(i))
            next
            self.state_series.append(next_state)

        self.predictions = [tf.nn.softmax(logits) for logits in self.state_series[1:]]
        self.losses = [tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
                  for logits, labels in zip(self.state_series[1:], labels_series)]

        # cross entropy loss
        # self.losses = [tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits), axis=1))
        #           for logits, labels in zip(self.predictions, labels_series)]

        # alternate - loop form
        # If using this loop, do not unstack inputs and labels
        # inputs will be split along first axis - permute axes
        # states = tf.scan(self.fn, tf.transpose(self.x, [1,0,2]), initializer=self.init_state)
        # self.state_series = tf.unstack(states, axis=0)  # output of states leaves t on zero axis
        # self.predictions = [tf.nn.softmax(logits) for logits in self.state_series]
        # losses = [tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
        #           for logits, labels in zip(self.state_series, labels_series)]

        # the logits are the states

        self.total_loss = tf.reduce_mean(self.losses)
        self.train_op = tf.train.AdamOptimizer(.001).minimize(self.total_loss)
        self.kill_diagonal = tf.assign(self.W, tf.multiply(self.W, self.the_mid_be_hittin_tho))

    def fn(self, hprev, input):
        # W = tf.get_variable('hidden_weights')
        h = tf.tanh(tf.matmul(hprev, self.W) + input)
        return h

    def run_training(self, inputs, labels, batch_size=16):
        """
        :param inputs: n x t x d input matrix
        :param labels: n x t x d label matrix
        :return:
        """
        sess.run(self.train_iter.initializer, feed_dict={self.x: inputs, self.y: labels})
        k = 0
        total_loss = []
        while k < len(inputs):
            rem = len(inputs) - k
            end = min(batch_size, rem)
            # cur_inputs = inputs[k:k+end]
            # cur_labels = labels[k:k+end]
            init_state = np.zeros((min(rem, batch_size), STATE_SIZE))
            cur_inputs, cur_labels = sess.run(self.next_element)
            # print(cur_labels[:,0,:])
            feed_dict = {self.x: cur_inputs, self.y: cur_labels, self.init_state: init_state}
            weights, states, loss, t_loss, _ = self.sess.run([self.W, self.state_series, self.losses, self.total_loss,
                                                              self.train_op], feed_dict=feed_dict)
            # self.sess.run(self.kill_diagonal)
            # print(np.log(states[0]))
            k += batch_size
            # print(weights)
            # print(loss)
            if k % 5 == 0:
                total_loss.append(t_loss)

        return weights, np.array(total_loss)

    def run_test(self, inputs, labels):
        """Presumably only one example will be run at a time for simple visualization."""
        init_state = np.zeros((1, STATE_SIZE))
        feed_dict = {self.x: inputs, self.y: labels, self.init_state: init_state}
        weights, states, pred, loss, total_loss = self.sess.run([self.W, self.state_series, self.predictions,
                                                                 self.losses, self.total_loss], feed_dict=feed_dict)

        print('test label', labels[0,0,:])
        print('final state', states[-1])
        print('final prediction', pred[-1])

        f1 = plt.figure()
        # heatmap data must be (neurons x time)
        ax1 = sns.heatmap(weights, cmap='RdBu_r', center=0)
        plt.title('Weights')

        f2 = plt.figure()
        # plt.subplot(211)
        ax2 = sns.heatmap(np.concatenate(states, axis=0), cmap='RdBu_r', center=0)
        plt.title('States')

        # f3 = plt.figure()
        # plt.subplot(211)
        # ax3 = sns.heatmap(np.concatenate(pred, axis=0))
        # ax3 = sns.heatmap(pred[-1])
        # plt.title('Predictions')

        # print(labels)
        # print(pred)
        # print(loss)


# label_test()
# fxn_test()
mat = np.random.uniform(-1, 1, size=(50, 50))

# sns.heatmap(mat, cmap='RdBu_r', center=0)
# sns.heatmap(mat, cmap='coolwarm', center=0)
# sns.heatmap(mat, cmap=sns.cubehelix_palette())
# print(mat)
# plt.show()
with tf.Session() as sess:
    steps = 10
    rnn = hatRNN(sess, timesteps=steps)
    sess.run(tf.global_variables_initializer())

    # ix = np.array([9, 8, 4])
    ix = np.random.randint(low=0, high=STATE_SIZE, size=10*batch)
    # print(ix)
    # print(sess.run(rnn.W))
    losses = []
    W_trend = []
    inputs, labels = create_2d_inputs(ix, spread=3, dev=1, create_noise=False, intensity=.5, density=.5,
                                      equal_ex=True)
    inputs, labels = prepare_training_data(inputs, labels, steps)
    # print(labels.shape)
    inputs = np.concatenate([inputs] * 10, axis=0)
    labels = np.concatenate([labels] * 10, axis=0)
    # print(labels.shape)
    run = True
    if run:
        for epoch in range(n_epoch):
            weights, loss = rnn.run_training(inputs, labels, batch_size=batch)
            losses.append(loss)
            if (epoch + 1) % 20 == 0:
                W_trend.append(weights)

            if (epoch+1) % 100 == 0:
                print(epoch)

        loss_cat = np.concatenate(losses)
        # print(loss_cat)
        rnn.run_test(inputs[5:6, :, :], labels[5:6, :, :])
        loss_fig = plt.figure()
        loss_ax = plt.plot(loss_cat)
        plt.title('Losses')
        # print(loss_cat[-1])
        # loss_ax = plt.plot(np.squeeze(losses))

        # f2 = plt.figure()
        # for i in range(10):
        #     plt.subplot(2, 5, i+1)
        #     sns.heatmap(W_trend[i], cmap='RdBu_r', center=0)

        # ax2 = sns.heatmap(np.concatenate(states, axis=0), cmap='RdBu_r', center=0)
        # plt.title('States')

        plt.show()
        ### Note that the loss can never be zero, as the first input can never produce the right output when the input
        ### weights are the identity matrix.

    # writer = tf.summary.FileWriter("./gviz/test/2")
    # writer.add_graph(sess.graph)


