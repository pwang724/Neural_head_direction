import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from create_hat_inputs import velocityTraining
import time
np.random.seed(2)
tf.set_random_seed(2)


class movingHatRNN:
    """An RNN made to model hat attractor/Mexican hat network properties."""

    def __init__(self, sess, timesteps, state_size=10, vel_size=2, support_size=10,
                 save_name="moving-hat", optimize_pretrained_mat=False):
        self.state_size = state_size
        self.hidden_size = support_size + vel_size
        self.full_size = self.state_size + self.hidden_size
        self.x = tf.placeholder(tf.float32, [None, timesteps, self.full_size], name='input_placeholder')
        self.y = tf.placeholder(tf.float32, [None, timesteps, self.state_size], name='output_placeholder')
        self.init_state = tf.placeholder(tf.float32, [None, self.full_size], name='init_state')
        self.sess = sess
        self.timesteps = timesteps
        self.save_name = save_name

        train_dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        train_dataset = train_dataset.shuffle(int(1E6)).batch(tf.cast(batch, tf.int64)).repeat()  # inferred repeat
        self.train_iter = train_dataset.make_initializable_iterator()
        self.next_element = self.train_iter.get_next()

        labels_series = tf.unstack(self.y, axis=1)
        # weights from visible state to visible state - pretrained
        self.W_h = tf.get_variable("W_h", shape=[3, self.state_size, self.state_size])

        # weights from hidden neurons to state
        self.W_ba = tf.get_variable("W_ab", shape=[3, self.hidden_size, self.state_size])
        self.W_xb = tf.get_variable("W_bx", shape=[3, self.full_size, self.hidden_size])

        states = tf.scan(self.gru_fn, tf.transpose(self.x, [1,0,2]), initializer=self.init_state)
        self.state_series = tf.unstack(states, axis=0)  # output of states leaves t on zero axis - (t x d) matrices
        self.visible_states = [logits[:, :self.state_size] for logits in self.state_series]
        self.predictions = [tf.nn.softmax(logits) for logits in self.visible_states]
        # the logits are the state_size portion of the states
        self.losses = [tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
                  for logits, labels in zip(self.visible_states, labels_series)]

        self.total_loss = tf.reduce_mean(self.losses)
        if optimize_pretrained_mat:
            var_list = [self.W_h, self.W_ba, self.W_xb]
            self.save_name += "-fullopt"
        else:
            var_list = [self.W_ba, self.W_xb]
            self.save_name += "-partopt"


        # Optimizers: AdamOptimizer and RMSProp are about the same. AdamOptimizer will be used for all results.
        # Reducing the learning rate doesn't help much.
        self.train_op = tf.train.AdamOptimizer(.001).minimize(self.total_loss, var_list=var_list)
        # self.train_op = tf.train.RMSPropOptimizer(.001).minimize(self.total_loss, var_list=var_list)
        # self.train_op = tf.train.GradientDescentOptimizer(.001).minimize(self.total_loss, var_list=var_list)
        # self.train_op = tf.train.AdagradOptimizer(.001).minimize(self.total_loss, var_list=var_list)
        self.state_saver = tf.train.Saver({"W_h": self.W_h})
        self.support_saver = tf.train.Saver({"W_ba": self.W_ba, "W_xb": self.W_xb})


    def gru_fn(self, hprev, input):
        #  update gate
        _z = tf.concat([tf.matmul(hprev[:, :self.state_size], self.W_h[0]) +
                        tf.matmul(hprev[:, self.state_size:], self.W_ba[0]),
                        tf.matmul(hprev, self.W_xb[0])], axis=1)
        z = tf.sigmoid(input + _z)
        #  reset gate
        _r = tf.concat([tf.matmul(hprev[:, :self.state_size], self.W_h[1]) +
                        tf.matmul(hprev[:, self.state_size:], self.W_ba[1]),
                        tf.matmul(hprev, self.W_xb[1])], axis=1)
        r = tf.sigmoid(input + _r)
        #  intermediate
        st_reset = r * hprev
        _h = tf.concat([tf.matmul(st_reset[:, :self.state_size], self.W_h[2]) +
                   tf.matmul(st_reset[:, self.state_size:], self.W_ba[2]),
                   tf.matmul(st_reset, self.W_xb[2])], axis=1)
        h = tf.tanh(input + _h)
        # new state
        st = (1 - z) * h + (z * hprev)
        # st = z * h + (1 - z) * hprev
        return st

    def run_training(self, inputs, labels, pretrained_file, n_epoch=1e3, batch_size=16, zero_tanh_diag=False, zero_sig_diag=False):
        """
        :param inputs: n x t x d input matrix
        :param labels: n x t x d label matrix
        :return:
        """
        self.state_saver.restore(sess, pretrained_file)
        self.sess.run(self.train_iter.initializer, feed_dict={self.x: inputs, self.y: labels})
        total_loss = []
        st = time.perf_counter()
        for n in range(n_epoch):
            cur_inputs, cur_labels = sess.run(self.next_element)
            init_state = np.zeros((cur_inputs.shape[0], self.full_size))
            feed_dict = {self.x: cur_inputs, self.y: cur_labels, self.init_state: init_state}
            states, loss, t_loss, _ = self.sess.run([self.state_series, self.losses, self.total_loss,
                                                     self.train_op], feed_dict=feed_dict)
            if n % 20 == 0:
                total_loss.append(t_loss)

            if (n+1) % 100 == 0:
                print(n)

        state_save_path = self.state_saver.save(self.sess, "./save_files/" + self.save_name + "-state")
        support_save_path = self.support_saver.save(self.sess, "./save_files/" + self.save_name + "-support")
        print("Weights saved")
        print("Training time: {}".format(time.perf_counter() - st))
        return states, np.array(total_loss)

    def run_test(self, inputs, labels):
        """Only one example is input for visualization."""
        # self.state_saver.restore(sess, checkpoint)
        # f4 = plt.figure()
        # a=self.sess.run(self.W_h[2])
        # sns.heatmap(a)

        self.support_saver.restore(self.sess, "./save_files/" + self.save_name + "-support")
        self.state_saver.restore(self.sess, "./save_files/" + self.save_name + "-state")
        init_state = np.zeros((1, self.full_size))
        feed_dict = {self.x: inputs, self.y: labels, self.init_state: init_state}
        W_h, W_ba, W_xb, states, pred, loss, total_loss = self.sess.run([self.W_h, self.W_ba, self.W_xb,
                                                                         self.state_series, self.predictions,
                                                                         self.losses, self.total_loss],
                                                                        feed_dict=feed_dict)
        _W_update = np.concatenate([W_h[0], W_ba[0]], axis=0)
        W_update = np.concatenate([_W_update, W_xb[0]], axis=1)
        _W_reset = np.concatenate([W_h[1], W_ba[1]], axis=0)
        W_reset = np.concatenate([_W_reset, W_xb[1]], axis=1)
        _W_tanh = np.concatenate([W_h[2], W_ba[2]], axis=0)
        W_tanh = np.concatenate([_W_tanh, W_xb[2]], axis=1)
        print('test label', labels[0,0,:])
        print('final state', states[-1])
        print('final prediction', np.round(pred[-1], 5))

        f3, (ax3, ax4, ax5) = plt.subplots(3, 1)
        plt.sca(ax3)
        sns.heatmap(np.concatenate(states, axis=0), cmap='RdBu_r', center=0)
        plt.title('States')

        plt.sca(ax4)
        sns.heatmap(np.concatenate(pred, axis=0), cmap='RdBu_r', center=0)
        plt.title('Predictions')

        plt.sca(ax5)
        sns.heatmap(labels[0], cmap='RdBu_r', center=0)
        plt.title('Labels')

        f4, (ax6, ax7, ax8) = plt.subplots(1, 3)
        plt.sca(ax6)
        sns.heatmap(W_tanh, cmap='RdBu_r', center=0)
        plt.title("Tanh weights")
        plt.sca(ax7)
        sns.heatmap(W_update, cmap='RdBu_r', center=0)
        plt.title("Update sigmoid weights")
        plt.sca(ax8)
        sns.heatmap(W_reset, cmap='RdBu_r', center=0)
        plt.title("Reset sigmoid weights")

        print(inputs.shape, labels.shape)

        plt.figure()
        sns.heatmap(inputs[0])
        # f5, (ax7, ax8) = plt.subplots(2, 1)
        # plt.sca(ax7)
        # sns.heatmap(labels[0])
        # plt.title('Predictions')
        # plt.sca(ax8)
        # sns.heatmap(labels[0])
        # plt.title('Labels')



with tf.Session() as sess:
    STATE_SIZE = 10
    support_size = 20  # must be at least size of velocity options
    n_epoch = int(5e3)
    batch = 10
    steps = 20
    pretraining_file = "hatRNN"

    # create training data
    ix = np.random.randint(low=0, high=STATE_SIZE, size=10 * batch)
    data = velocityTraining(STATE_SIZE)
    inputs, labels, vel_size = data.create_inputs(ix, timesteps=steps, spread=3, velocity_start=5)
    # append zeros for desired full state size
    inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], inputs.shape[1], support_size))], axis=2)
    save_name = "movinghat-spread3"
    rnn = movingHatRNN(sess, steps, save_name=save_name, support_size=support_size, optimize_pretrained_mat=True)
    sess.run(tf.global_variables_initializer())

    run = False
    if run:
        weights, loss = rnn.run_training(inputs, labels, pretrained_file=pretraining_file,
                                         n_epoch=n_epoch, batch_size=batch)
        # print(loss.shape)
        pd.DataFrame(loss).to_csv("./save_files/" + save_name + "-loss.csv")

    test = True
    if test:
        loss_frame = pd.read_csv("./save_files/" + save_name + "-loss.csv")
        loss = loss_frame.as_matrix()[:, 1]
        rnn.run_test(inputs[0:1, :, :], labels[0:1, :, :])

    if run or test:
        loss_fig = plt.figure()
        plt.title('Losses')
        loss_ax = plt.plot(loss)

    compare_loss = False
    if compare_loss:
        adam = pd.read_csv("./save_files/" + save_name + "-Adamloss").as_matrix()[:, 1]
        sgd = pd.read_csv("./save_files/" + save_name + "-SGDloss").as_matrix()[:, 1]
        ada = pd.read_csv("./save_files/" + save_name + "-Adaloss").as_matrix()[:, 1]
        rms = pd.read_csv("./save_files/" + save_name + "-RMSloss").as_matrix()[:, 1]

        losses = np.stack([adam, sgd, ada, rms], axis=1)
        loss_comp, loss_comp_ax = plt.subplots()
        plt.title('Loss comparison')
        plt.plot(adam, label='Adam')
        plt.plot(sgd, label='SGD')
        plt.plot(ada, label='AdaGrad')
        plt.plot(rms, label='RMSProp')
        loss_comp_ax.legend()

    plt.show()
    ### Note that the loss can never be zero, as the first input can never produce the right output when the input
    ### weights are the identity matrix.

