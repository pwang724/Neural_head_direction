import numpy as np
import tensorflow as tf
import model
from config import Options
from datasets import inputs as inputs_helper
import utils
# import matplotlib as mpl
# mpl.use('TkAgg')
import time


def run_training(opts):
    inputs, labels, angle_trig, opts = inputs_helper.make_inputs(opts)
    rnn = model.RNN(opts)
    train_iter, next_element = rnn.batch_inputs(inputs, labels, opts)
    t = time.perf_counter()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        n_epoch = opts.n_epoch
        n_batch = int(opts.n_examples / opts.batch_size)
        losses = []
        lr = np.logspace(-3, -4, n_epoch)  # learning rate
        for epoch in range(n_epoch):
            sess.run(train_iter.initializer)
            for batch in range(n_batch):
                x, y = sess.run(next_element)
                init_state = np.zeros((y.shape[0], opts.network_state_size))
                feed_dict = {rnn.inputs: x, rnn.labels: y, rnn.init_state: init_state, rnn.lr: lr[epoch]}
                pred, loss, _ = sess.run([rnn.predictions, rnn.total_loss, rnn.train_op], feed_dict=feed_dict)
                losses.append(loss)

            if (epoch + 1) % 100 == 0:
                print('Epoch {0}, time elapsed: {1:.2f}s'.format(epoch, time.perf_counter() - t))

        W_ah, W_sh, W_hh, W_out, bias = sess.run([rnn.W_ah_trained, rnn.W_sh_trained, rnn.W_hh_masked, rnn.W_out, rnn.bias])

    # save weights
    opts.weights_name = opts.get_path() + "_weights"
    utils.save_weights([W_ah, W_sh, W_hh, W_out, bias], opts.folder, opts.weights_name)
    tup = [('Losses', np.array(losses))]
    fig_name = opts.folder + '/' + opts.get_path() + '_loss_fig'
    utils.pretty_plot(tup, 1, 1, fig_name)
    return opts


def run_test(opts):
    rnn = model.RNN(opts)
    inputs, labels, angle_trig, opts = inputs_helper.make_inputs(opts)
    train_iter, next_element = rnn.batch_inputs(inputs, labels, opts)
    pred, states = [], []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_iter.initializer)
        run = True
        while run:
            try:
                x, y = sess.run(next_element)
                init_state = np.zeros((y.shape[0], opts.network_state_size))
                feed_dict = {rnn.inputs: x, rnn.labels: y, rnn.init_state: init_state}
                p, s = sess.run([rnn.predictions, rnn.states], feed_dict=feed_dict)  # p is arrays in time
                pred.append(np.stack(p, axis=1))
                states.append(np.transpose(s, [1, 0, 2]))
            except tf.errors.OutOfRangeError:  # done iterating
                run = False

        pred = np.concatenate(pred, axis=0)
        states = np.concatenate(states, axis=0)

    return inputs, labels, pred, states, angle_trig, opts


if __name__ == '__main__':
    opts = Options()
    opts.weights_name = run_training(opts)
    opts.n_examples = opts.batch_size
    run_test(opts)