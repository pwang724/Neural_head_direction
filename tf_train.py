import tensorflow as tf
import numpy as np
from datasets import inputs
import os, time
import pickle as pkl
import utils as utils
import config
from tf_rnn import RNN
from collections import defaultdict

# set seed for reproducibility
# np.random.seed(2)
# tf.set_random_seed(2)

def create_tf_dataset(x, y, noise, batch_size, shuffle=True):
    data = tf.data.Dataset.from_tensor_slices((x, y, noise))
    if shuffle:
        data = data.shuffle(int(1E6))
    data = data.batch(tf.cast(batch_size, tf.int64)).repeat()
    train_iter = data.make_initializable_iterator()
    next_element = train_iter.get_next()
    return train_iter, next_element

def create_placeholders(d_in, d_out, layers, T):
    x = tf.placeholder(tf.float32, [None, T, d_in], name='input_placeholder')
    y = tf.placeholder(tf.float32, [None, T, d_out], name='output_placeholder')
    n = tf.placeholder(tf.float32, [None, T, layers], name='output_placeholder')
    return x, y, n

def modify_path(path):
    n = 0
    add_on = '_' + format(n, '02d')
    path_mod = path + add_on
    while (os.path.exists(path_mod)):
        n += 1
        add_on = '_' + format(n, '02d')
        path_mod = path + add_on
    os.makedirs(path_mod)
    return path_mod

def save_activity(model, x, y, noise, path=None, save_name=None):
    # run some test activity
    if path is not None:
        save_path = path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = model.save_path

    if save_name is None:
        save_name = model.opts.activity_name

    f_name = os.path.join(save_path, save_name)
    sess = tf.get_default_session()
    states, predictions, loss = sess.run([model.states, model.predictions, model.error_loss])
    data = {'X': x, 'Y': y, 'N': noise, 'states': states, 'predictions': predictions, 'loss': loss}
    with open(f_name + ".pkl", 'wb') as f:
        pkl.dump(data, f)
    print('activity saved in ' + f_name)


def train(opts, seed=False):
    """
    :param inputs: n x t x d input matrix
    :param labels: n x t x d label matrix
    :return:
    """
    n_epoch = opts.epoch
    print('n epoch', n_epoch)
    save_path = opts.save_path
    n_batch_per_epoch = opts.n_input // opts.batch_size
    tf.reset_default_graph()
    # with tf.Graph().as_default() as graph:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if seed:
        seednum = 2
        np.random.seed(seednum)
        tf.set_random_seed(seednum)

    X, Y, N, _ = inputs.create_inputs(opts)
    X_pl, Y_pl, N_pl = create_placeholders(X.shape[-1], Y.shape[-1], opts.rnn_size, X.shape[1])
    train_iter, next_element = create_tf_dataset(X_pl, Y_pl, N_pl, opts.batch_size)
    model = RNN(next_element, opts, training=True)

    logger = defaultdict(list)  # return an empty list for keys not present, set those keys to a value of empty list
    print('starting')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(train_iter.initializer, feed_dict={X_pl: X, Y_pl: Y, N_pl: N})
        if opts.load_checkpoint:
            model.load()

        t = time.perf_counter()
        for ep in range(n_epoch):
            for b in range(n_batch_per_epoch):
                cur_loss, error_loss, weight_loss, activity_loss, states, _ = sess.run(
                    [model.total_loss, model.error_loss, model.weight_loss,
                     model.activity_loss, model.states, model.train_op])
                # grads_and_vars = sess.run(model.grad)[0]
                # grad, var = grads_and_vars
                # print(grad)
                # print(Whh,'\n',Wxh,'\n',Wout)
                assert not np.isnan(error_loss), "Error is NaN, retry"

            if (ep % 1 == 0 and ep>0):  # save to loss file
                logger['epoch'] = ep
                logger['loss'].append(cur_loss)
                logger['error_loss'].append(error_loss)
                logger['activity_loss'].append(activity_loss)
                logger['weight_loss'].append(weight_loss)

            # if (ep+1) % 25 == 0: #display in terminal
            # if (ep+1) % 10 == 0:
            #     print('[*] Epoch %d  total_loss=%.2f error_loss=%.2f a_loss=%.2f, w_loss=%.2f'
            #           % (ep+1, cur_loss, error_loss, activity_loss, weight_loss))
            print('[*] Epoch %d  total_loss=%.2f error_loss=%.2f a_loss=%.2f, w_loss=%.2f'
                  % (ep, cur_loss, error_loss, activity_loss, weight_loss))
            # Whh, Wxh, Wout = sess.run([model.Whh, model.Wxh, model.Wout])
            # print(Whh)
            # print(states[0,:10,:])
            tnew = time.perf_counter()
            print(f'{tnew - t} seconds elapsed')
            t = tnew

        model.save(save_path)
        model.save_weights(save_path)
        with open(os.path.join(save_path, opts.log_name + '.pkl'), 'wb') as f:
            pkl.dump(logger, f)

    data = {'X': X, 'Y': Y, 'N': N}
    train_path = os.path.join(save_path, 'training_set.pkl')
    with open(train_path, 'wb') as f:
        pkl.dump(data, f)
    save_name = os.path.join(save_path, opts.parameter_name)
    utils.save_parameters(opts, save_name)
    return opts, save_name


def eval(opts, data=None):
    # generate and evaluate a test set for analysis
    print('eval start')
    save_path = opts.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('graph start')
    tf.reset_default_graph()
    if data:
        X, Y, N, vel = data
    else:
        X, Y, N, vel = inputs.create_inputs(opts)

    opts.n_inputs = X.shape[0]
    opts.batch_size = opts.n_inputs
    X_pl, Y_pl, N_pl = create_placeholders(X.shape[-1], Y.shape[-1], opts.rnn_size, X.shape[1])
    train_iter, next_element = create_tf_dataset(X_pl, Y_pl, N_pl, opts.batch_size, shuffle=False)

    print('rnn start')
    model = RNN(next_element, opts, training=False)

    save_name = opts.activity_name
    print('[*] Testing')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(train_iter.initializer, feed_dict={X_pl: X, Y_pl: Y, N_pl: N})
        # sess.run(train_iter.initializer, feed_dict={X_pl: X, Y_pl: Y})
        print('loading saved')
        model.load()
        save_activity(model, X, Y, N, save_path, save_name)


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
    X_pl, Y_pl, N_pl = create_placeholders(X.shape[-1], Y.shape[-1], opts.rnn_size, X.shape[1])
    train_iter, next_element = create_tf_dataset(X_pl, Y_pl, N_pl, opts.batch_size, shuffle=False)
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
        save_activity(model, X, Y, N, save_path, group + '_silence')



if __name__ == '__main__':
    st_model_opts = config.Options()
    opts = st_model_opts
    # opts.save_path = './t/'
    opts.epoch = 300
    opts.load_checkpoint = False
    train(opts)
