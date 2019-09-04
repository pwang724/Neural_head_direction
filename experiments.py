import tensorflow as tf
import numpy as np
import os
import pickle as pkl
import config
from tf_rnn import RNN
import tf_train as train


def silence(opts, group, data=None):
    """Silence a neuron by setting its input and output weights to zero."""
    print('start')
    save_path = opts.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('graph start')
    tf.reset_default_graph()
    # if data:
    #     X, Y, N, vel = data
    # else:
    #     X, Y, N, vel = inputs.create_inputs(opts)
    with open(os.path.join(save_path, 'moving_activity.pkl'), 'rb') as f:
        act = pkl.load(f)
        X, Y, N = act['X'], act['Y'], act['N']

    with open(os.path.join(save_path, 'ix_dict.pkl'), 'rb') as f:
        ix_dict = pkl.load(f)

    opts.n_inputs = X.shape[0]
    opts.batch_size = opts.n_inputs
    X_pl, Y_pl, N_pl = train.create_placeholders(X.shape[-1], Y.shape[-1], opts.rnn_size, X.shape[1])
    train_iter, next_element = train.create_tf_dataset(X_pl, Y_pl, N_pl, opts.batch_size, shuffle=False)
    lesion_ix = ix_dict[group]

    print('rnn start')
    # model = RNN(next_element, opts, training=False)
    model = RNN(next_element, opts, training=False, lesion_ix=lesion_ix)

    print('[*] Testing')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(train_iter.initializer, feed_dict={X_pl: X, Y_pl: Y, N_pl: N})
        # sess.run(train_iter.initializer, feed_dict={X_pl: X, Y_pl: Y})
        print('loading saved')
        model.load()

        # assign silenced weights
        Whh = sess.run(model._Whh)
        # Whh[:, lesion_ix] = 0
        # Whh[lesion_ix, :] = 0
        # sess.run(tf.assign(model._Whh, Whh))
        train.save_activity(model, X, Y, N, save_path, group + '_silence')


def posn_blast(opts):
    """Blast a single position with enough activity to activate its shifter neurons without a velocity input.
    Use purely zero inputs, and blast a single position."""
    save_path = opts.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('graph start')
    tf.reset_default_graph()
    with open(os.path.join(save_path, 'moving_activity.pkl'), 'rb') as f:
        act = pkl.load(f)
        X, Y, N = act['X'], act['Y'], act['N']

    with open(os.path.join(save_path, 'ix_dict.pkl'), 'rb') as f:
        ix_dict = pkl.load(f)

    # just use the first position for simplicity
    Ering = ix_dict['E_ring']
    T = 20
    n_ipt = len(Ering)
    blast = np.zeros((n_ipt, T, opts.rnn_size))
    blast[np.arange(n_ipt), 5, Ering] = 100  # one strong blast at T=5

    X = np.zeros_like(X)[:n_ipt, :T]
    Y = np.zeros_like(Y)[:n_ipt, :T]
    N = np.zeros_like(N)[:n_ipt, :T]

    opts.n_inputs = X.shape[0]
    opts.batch_size = opts.n_inputs
    X_pl, Y_pl, N_pl = train.create_placeholders(X.shape[-1], Y.shape[-1], opts.rnn_size, X.shape[1])
    train_iter, next_element = train.create_tf_dataset(X_pl, Y_pl, N_pl, opts.batch_size, shuffle=False)

    print('rnn start')
    # model = RNN(next_element, opts, training=False)
    model = RNN(next_element, opts, training=False, perturb=blast)

    print('[*] Testing')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(train_iter.initializer, feed_dict={X_pl: X, Y_pl: Y, N_pl: N})
        print('loading saved')
        model.load()

        # assign silenced weights
        train.save_activity(model, X, Y, N, save_path, 'ring_blast')


def posn_bump(opts):
    """Input a velocity signal while inhibiting the ring, then add a bit of current to one neuron."""
    pass


if __name__ == '__main__':
    st_model_opts = config.Options()
    opts = st_model_opts
    # opts.save_path = './t/'
    opts.epoch = 300
    opts.load_checkpoint = False
    train(opts)
