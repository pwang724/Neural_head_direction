from collections import namedtuple
import numpy as np
import tensorflow as tf
import pickle as pkl
import matplotlib.pylab as plt
from PIL import Image

def define_nonstationary_weights(opts):
    state_size = opts.state_size
    velocity_size = opts.velocity_size
    rnn_size = opts.rnn_size

    #weights
    weight_struct = namedtuple("weight_struct", "W_i_a W_i_b W_h W_h_aa W_h_ba W_h_ab_bb W_h_mask W_out W_b W_h_z W_h_r")
    k = weight_struct(W_i_a='W_i_a', W_i_b='W_i_b',
                      W_h_aa= 'W_h_aa', W_h_ba= 'W_h_ba', W_h_ab_bb= 'W_h_ab_bb', W_b='W_b',
                      W_h_mask='W_h_mask', W_h= 'W_h',
                      W_h_z = 'W_h_z', W_h_r = 'W_h_r',
                      W_out='W_out')

    with tf.variable_scope('input', reuse= tf.AUTO_REUSE):
        support_size = rnn_size - state_size
        W_i_a = tf.constant(np.eye(state_size), dtype=tf.float32)
        W_i_b = tf.get_variable(k.W_i_b, shape=[velocity_size, support_size])

    with tf.variable_scope('output', reuse= tf.AUTO_REUSE):
        W_out_np = np.zeros((rnn_size, state_size))
        np.fill_diagonal(W_out_np, 1)
        W_out = tf.constant(W_out_np, dtype=tf.float32)

    with tf.variable_scope('hidden', reuse= tf.AUTO_REUSE):
        W_h_mask = tf.get_variable(k.W_h_mask, shape=[rnn_size, rnn_size],
                                   dtype=tf.float32, trainable=False)
        W_h_aa = tf.get_variable(k.W_h_aa, shape=[state_size, state_size])
        W_h_ba = tf.get_variable(k.W_h_ba, shape=[support_size, state_size])
        W_h_ab_bb = tf.get_variable(k.W_h_ab_bb, shape=[rnn_size, support_size])
        W_b = tf.get_variable(k.W_b, shape=[rnn_size], initializer=tf.constant_initializer(0.0))
        W_h_left = tf.concat([W_h_aa, W_h_ba], axis=0)
        W_h = tf.concat([W_h_left, W_h_ab_bb], axis=1)

    weight_dict = {k.W_i_a: W_i_a, k.W_i_b: W_i_b,
                   k.W_h_mask: W_h_mask, k.W_h: W_h,
                   k.W_h_aa: W_h_aa, k.W_h_ba: W_h_ba, k.W_h_ab_bb: W_h_ab_bb, k.W_b: W_b,
                   k.W_out: W_out}
    return k, weight_dict


def define_stationary_weights(opts):
    state_size = opts.state_size
    rnn_size = opts.rnn_size
    # weights
    weight_struct = namedtuple("weight_struct", "W_in W_h W_h_mask W_out W_b")
    k = weight_struct(W_in='W_in', W_h='W_h', W_h_mask='W_h_mask', W_out='W_out', W_b='W_b')

    #input weights
    with tf.variable_scope('input', reuse=tf.AUTO_REUSE):
        W_in_np = np.zeros((state_size, rnn_size))
        np.fill_diagonal(W_in_np, 1)
        W_in_tf = tf.constant(W_in_np, dtype=tf.float32, name=k.W_in)

    with tf.variable_scope('output', reuse=tf.AUTO_REUSE):
        W_out_np = np.zeros((rnn_size, state_size))
        np.fill_diagonal(W_out_np, 1)
        W_out_tf = tf.constant(W_out_np, dtype=tf.float32, name=k.W_out)

    with tf.variable_scope('hidden', reuse=tf.AUTO_REUSE):
        W_h_mask_tf = tf.get_variable(k.W_h_mask, shape=[rnn_size, rnn_size], dtype=tf.float32, trainable=False)
        W_h_tf = tf.get_variable(k.W_h, shape=[rnn_size, rnn_size])
        W_b_tf = tf.get_variable(k.W_b, shape=[rnn_size], initializer=tf.constant_initializer(0.0))

    weight_dict = {k.W_in: W_in_tf, k.W_out: W_out_tf, k.W_h: W_h_tf, k.W_h_mask: W_h_mask_tf, k.W_b: W_b_tf}
    return k, weight_dict

def rnn_non_stationary(weight_dict, k, h_prev, input, name, opts):
    state_size = opts.state_size
    W_i_a = weight_dict[k.W_i_a]
    W_i_b = weight_dict[k.W_i_b]
    W_h = weight_dict[k.W_h]
    W_h_mask = weight_dict[k.W_h_mask]
    W_b = weight_dict[k.W_b]

    in_a = tf.matmul(input[:,:state_size], W_i_a)
    in_b = tf.matmul(input[:,state_size:], W_i_b)
    a_in = tf.concat([in_a, in_b], axis= 1)

    W_h = tf.multiply(W_h, W_h_mask)
    # regular rnn
    W_h_masked = tf.multiply(W_h, W_h_mask)
    a_hidden = tf.matmul(h_prev, W_h_masked)
    out = tf.tanh(W_b + a_in + a_hidden, name='time_{}'.format(name))

    # #gru
    # W_h_z = weight_dict[k.W_h_z]
    # W_h_r = weight_dict[k.W_h_r]
    # W_h_z = tf.multiply(W_h_z, W_h_mask)
    # W_h_r = tf.multiply(W_h_r, W_h_mask)
    # z = tf.sigmoid(a_in + tf.matmul(h_prev, W_h_z))
    # r = tf.sigmoid(a_in + tf.matmul(h_prev, W_h_r))
    # hprime = tf.tanh(a_in + tf.multiply(tf.matmul(h_prev, W_h), r) + W_b)
    # out = tf.add(tf.multiply((1-z), h_prev), tf.multiply(z, hprime), name='time_{}'.format(name))
    return out

def rnn_stationary(weight_dict, k, h_prev, input, name, opts):
    W_h = weight_dict[k.W_h]
    W_h_mask = weight_dict[k.W_h_mask]
    W_in = weight_dict[k.W_in]
    W_b = weight_dict[k.W_b]

    a_in = tf.matmul(input, W_in)

    W_h_masked = tf.multiply(W_h, W_h_mask)
    a_hidden = tf.matmul(h_prev, W_h_masked)

    out = tf.tanh(W_b + a_in + a_hidden, name='time_{}'.format(name))
    return out

def initialize_nonstationary_weights(sess, opts, weight_dict, k):
    state_size = opts.state_size
    rnn_size = opts.rnn_size
    load_weights = opts.load_weights
    dir_weights = opts.dir_weights

    W_h_aa_tf = weight_dict[k.W_h_aa]
    W_h_ba_tf = weight_dict[k.W_h_ba]
    W_h_ab_bb_tf = weight_dict[k.W_h_ab_bb]
    W_h_mask_tf = weight_dict[k.W_h_mask]
    W_b_tf = weight_dict[k.W_b]

    if load_weights:
        with open(dir_weights, 'rb') as f:
            w_dict = pkl.load(f)
        #both keys refers to the W_h of the stationary weight. therefore, cannot use k to reference
        W_h_old = w_dict['model/hidden/W_h:0']
        W_b_old = w_dict['model/hidden/W_b:0']
        old_W_aa = W_h_old[:state_size, :state_size]
        old_W_aa_filled = np.copy(old_W_aa)
        np.fill_diagonal(old_W_aa_filled, 1)

        resized = np.zeros((rnn_size-state_size, state_size))
        r = np.arange(0, rnn_size-state_size, (rnn_size-state_size)/state_size)
        for i in range(rnn_size-state_size):
            ix = (np.fabs(r - i)).argmin()
            resized[i,:] = old_W_aa_filled[ix,:]
        stretched_ba = resized
        stretched_ab = resized.transpose()

        W_h_aa = W_h_aa_tf.eval()
        W_h_ba = W_h_ba_tf.eval()
        W_h_ab_bb = W_h_ab_bb_tf.eval()
        bigmat = np.hstack((np.vstack((W_h_aa, W_h_ba)), W_h_ab_bb))
        bigmat[:len(W_h_old), :len(W_h_old)] = W_h_old
        # bigmat[state_size:, :state_size] = stretched_ba
        # bigmat[:state_size, state_size:] = stretched_ab

        sess.run(tf.assign(W_h_aa_tf, bigmat[:state_size,:state_size]))
        sess.run(tf.assign(W_h_ba_tf, bigmat[state_size:,:state_size]))
        sess.run(tf.assign(W_h_ab_bb_tf, bigmat[:,state_size:]))

        W_b = W_b_tf.eval()
        W_b[:len(W_b_old)] = W_b_old
        sess.run(tf.assign(W_b_tf, W_b))
    else:
        W_h_aa = W_h_aa_tf.eval()
        np.fill_diagonal(W_h_aa, 0)
        sess.run(tf.assign(W_h_aa_tf, W_h_aa))

    W_h_mask = W_h_mask_tf.eval()
    W_h_mask[:,:] = 1
    np.fill_diagonal(W_h_mask, 0)
    W_h_mask[state_size:,state_size:] = 0
    sess.run(tf.assign(W_h_mask_tf, W_h_mask))

def initialize_stationary_weights(sess, opt, weight_dict, k):
    state_size = opt.state_size
    rnn_size = opt.rnn_size
    W_h_mask_tf = weight_dict[k.W_h_mask]
    W_h_tf = weight_dict[k.W_h]

    W_h_mask = W_h_mask_tf.eval()
    W_h_mask[:, :] = 1
    W_h_mask[state_size:, state_size:] = 0
    np.fill_diagonal(W_h_mask, 0)

    W_h = W_h_tf.eval()
    np.fill_diagonal(W_h, 0)
    W_h[state_size:, state_size:] = 0

    sess.run(tf.assign(W_h_mask_tf, W_h_mask))
    sess.run(tf.assign(W_h_tf, W_h))
