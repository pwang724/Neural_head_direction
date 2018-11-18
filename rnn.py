from collections import namedtuple
import numpy as np
import tensorflow as tf
import pickle as pkl
import matplotlib.pylab as plt
import config
import os
import utils


# set seed for reproducibility
np.random.seed(2)
tf.set_random_seed(2)

def get_tf_vars_as_dict():
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    var_dict = {os.path.split(v.name)[1][:-2]: v for v in vars}
    return var_dict

def define_weights(opts):
    k = config.weight_names()
    state_size = opts.state_size
    rnn_size = opts.rnn_size
    support_size = rnn_size - state_size

    if opts.stationary:
        with tf.variable_scope('input'):
            tf.get_variable(k.W_in, shape=[state_size, rnn_size], trainable=False)
    else:
        with tf.variable_scope('input'):
            velocity_size = opts.velocity_size
            tf.get_variable(k.W_i_a, shape=[state_size, state_size], trainable=False)
            tf.get_variable(k.W_i_b, shape=[velocity_size, support_size])

    with tf.variable_scope('output', reuse= tf.AUTO_REUSE):
        tf.get_variable(k.W_out, shape=[rnn_size, state_size], trainable=False)

    with tf.variable_scope('hidden', reuse= tf.AUTO_REUSE):
        tf.get_variable(k.W_h_mask, shape=[rnn_size, rnn_size], trainable=False)
        tf.get_variable(k.W_b, shape=[rnn_size], initializer=tf.constant_initializer(0.0))
        W_h_aa = tf.get_variable(k.W_h_aa, shape=[state_size, state_size])
        W_h_ba = tf.get_variable(k.W_h_ba, shape=[support_size, state_size])
        W_h_ab = tf.get_variable(k.W_h_ab, shape=[state_size, support_size])
        W_h_bb = tf.get_variable(k.W_h_bb, shape=[support_size, support_size])
        W_h_left = tf.concat([W_h_aa, W_h_ba], axis=0)
        W_h_right = tf.concat([W_h_ab, W_h_bb], axis=0)
        W_h = tf.concat([W_h_left, W_h_right], axis=1)
    return W_h

def rnn(h_prev, input, name, opts):
    k = config.weight_names()
    weight_dict = get_tf_vars_as_dict()
    state_size = opts.state_size
    W_h_aa = weight_dict[k.W_h_aa]
    W_h_ab = weight_dict[k.W_h_ab]
    W_h_ba = weight_dict[k.W_h_ba]
    W_h_bb = weight_dict[k.W_h_bb]
    W_h_left = tf.concat([W_h_aa, W_h_ba], axis=0)
    W_h_right = tf.concat([W_h_ab, W_h_bb], axis=0)
    W_h = tf.concat([W_h_left, W_h_right], axis=1)

    W_h_mask = weight_dict[k.W_h_mask]
    W_b = weight_dict[k.W_b]
    W_out = weight_dict[k.W_out]

    if opts.stationary:
        W_in = weight_dict[k.W_in]
        a_in = tf.matmul(input, W_in)
    else:
        W_i_a = weight_dict[k.W_i_a]
        W_i_b = weight_dict[k.W_i_b]
        in_a = tf.matmul(input[:, :state_size], W_i_a)
        in_b = tf.matmul(input[:, state_size:], W_i_b)
        a_in = tf.concat([in_a, in_b], axis=1)

    W_h_masked = tf.multiply(W_h, W_h_mask)
    a_hidden = tf.matmul(h_prev, W_h_masked)
    state = tf.tanh(W_b + a_in + a_hidden, name='time_{}'.format(name))
    logit = tf.matmul(state, W_out)
    return state, logit

def initialize_weights(opts):
    sess = tf.get_default_session()
    k = config.weight_names()
    weight_dict = get_tf_vars_as_dict()

    state_size = opts.state_size
    rnn_size = opts.rnn_size
    support_size = rnn_size - state_size

    W_h_aa_tf = weight_dict[k.W_h_aa]
    W_h_ba_tf = weight_dict[k.W_h_ba]
    W_h_ab_tf = weight_dict[k.W_h_ab]
    W_h_bb_tf = weight_dict[k.W_h_bb]
    W_h_mask_tf = weight_dict[k.W_h_mask]
    W_b_tf = weight_dict[k.W_b]
    W_out_tf = weight_dict[k.W_out]

    sess.run(tf.assign(W_out_tf, np.eye(rnn_size, state_size)))
    sess.run(tf.assign(W_h_bb_tf, np.zeros((support_size, support_size))))
    W_h_mask = W_h_mask_tf.eval()
    W_h_mask[:, :] = 1
    np.fill_diagonal(W_h_mask, 0)
    if opts.mask_Wbb:
        W_h_mask[state_size:, state_size:] = 0
    sess.run(tf.assign(W_h_mask_tf, W_h_mask))

    if opts.stationary:
        W_in_tf = weight_dict[k.W_in]
        sess.run(tf.assign(W_in_tf, np.eye(state_size, rnn_size)))

        W_h_aa = W_h_aa_tf.eval()
        np.fill_diagonal(W_h_aa, 0)
        sess.run(tf.assign(W_h_aa_tf, W_h_aa))
        print('[!!!] Stationary weights initialized')
    else:
        W_i_a_tf = weight_dict[k.W_i_a]
        sess.run(tf.assign(W_i_a_tf, np.eye(state_size, state_size)))

        if opts.load_weights:
            with open(opts.dir_weights, 'rb') as f:
                w_dict = pkl.load(f)
                # cannot use k to reference old weights stored in file
                W_h_aa_old = w_dict['model/hidden/W_h_aa:0']
                W_h_bb_old = w_dict['model/hidden/W_h_bb:0']
                W_h_ab_old = w_dict['model/hidden/W_h_ab:0']
                W_h_ba_old = w_dict['model/hidden/W_h_ba:0']
                W_b_old = w_dict['model/hidden/W_b:0']

                # fill with stretch
                # W_h_aa_old_filled = np.copy(W_h_aa_old)
                W_h_aa_old_filled = np.eye(W_h_aa_old.shape[0]) * 0.25
                np.fill_diagonal(W_h_aa_old_filled, 1)
                resized = np.zeros((rnn_size-state_size, state_size))
                r = np.arange(0, rnn_size-state_size, (rnn_size-state_size)/state_size)
                for i in range(rnn_size-state_size):
                    ix = (np.fabs(r - i)).argmin()
                    resized[i,:] = W_h_aa_old_filled[ix,:]
                if opts.initialize_W_ab_diagonal:
                    W_h_ab_old = resized.transpose()
                if opts.initialize_W_ba_diagonal:
                    W_h_ba_old = resized

                sess.run(tf.assign(W_h_aa_tf, W_h_aa_old))
                sess.run(tf.assign(W_h_ba_tf, W_h_ba_old))
                sess.run(tf.assign(W_h_ab_tf, W_h_ab_old))
                sess.run(tf.assign(W_h_bb_tf, W_h_bb_old))

            W_b = W_b_tf.eval()
            W_b[:len(W_b_old)] = W_b_old
            sess.run(tf.assign(W_b_tf, W_b))
            print('[!!!] Moving weights initialized from ' + opts.dir_weights)
        else:
            W_h_aa = W_h_aa_tf.eval()
            np.fill_diagonal(W_h_aa, 0)
            sess.run(tf.assign(W_h_aa_tf, W_h_aa))
            print('[!!!] Moving weights initialized without pre-loading from stationary')

    if opts.debug_weights:
        for n, w_tf in weight_dict.items():
            w = w_tf.eval()
            if np.ndim(w) == 0:
                print(n, w)
            else:
                if np.ndim(w) == 1:
                    w = w.reshape(1,-1)
                plt.title(n)
                plt.imshow(w)
                plt.show()
