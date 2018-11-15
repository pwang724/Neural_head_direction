import tensorflow as tf
import numpy as np
import inputs
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle as pkl

import utils as utils
import rnn as rnn_helper
import config
from model import RNN
from collections import defaultdict

# set seed for reproducibility
# np.random.seed(2)
# tf.set_random_seed(2)

def create_tf_dataset(x, y, batch_size):
    data = tf.data.Dataset.from_tensor_slices((x, y))
    data = data.shuffle(int(1E6)).batch(tf.cast(batch_size, tf.int64)).repeat()
    train_iter = data.make_initializable_iterator()
    next_element = train_iter.get_next()
    return train_iter, next_element

def create_placeholders(opts):
    stationary = opts.stationary
    time_steps = opts.time_steps
    state_size = opts.state_size
    if stationary:
        x = tf.placeholder(tf.float32, [None, time_steps, state_size], name='input_placeholder')
    else:
        velocity_size = opts.velocity_max * 2
        x = tf.placeholder(tf.float32, [None, time_steps, state_size + velocity_size],
                           name='input_placeholder')
    y = tf.placeholder(tf.float32, [None, time_steps, state_size], name='output_placeholder')
    return x, y

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

def train(opts):
    """
    :param inputs: n x t x d input matrix
    :param labels: n x t x d label matrix
    :return:
    """
    n_epoch = opts.epoch
    save_path = opts.save_path
    file_name = opts.file_name
    stationary = opts.stationary
    n_batch_per_epoch = opts.n_input // opts.batch_size

    with tf.Graph().as_default() as graph:
        if not os.path.exists(opts.save_path):
            os.makedirs(opts.save_path)

        X, Y = inputs.create_inputs(opts)
        train_iter, next_element = create_tf_dataset(X, Y, opts.batch_size)
        model = RNN(next_element[0], next_element[1], opts, training=True)

        logger = defaultdict(list)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(train_iter.initializer)

            if opts.load_checkpoint:
                model.load()
            else:
                if stationary:
                    rnn_helper.initialize_stationary_weights(sess,
                                                             opts, model.weight_dict, model.k)
                else:
                    rnn_helper.initialize_nonstationary_weights(sess,
                                                                opts, model.weight_dict, model.k)

            for ep in range(n_epoch):
                for b in range(n_batch_per_epoch):
                    cur_loss, xe_loss, weight_loss, activity_loss, _ = sess.run(
                        [model.total_loss, model.xe_loss, model.weight_loss,
                         model.activity_loss, model.train_op])

                if (ep % 1 == 0 and ep>0): #save to loss file
                    logger['epoch'] = ep
                    logger['loss'].append(cur_loss)
                    logger['xe_loss'].append(xe_loss)
                    logger['activity_loss'].append(activity_loss)
                    logger['weight_loss'].append(weight_loss)
                if (ep % 5 == 0 and ep>0): #display in terminal
                    print('[*] Epoch %d  total_loss=%.2f xe_loss=%.2f a_loss=%.2f, w_loss=%.2f'
                          % (ep, cur_loss, xe_loss, activity_loss, weight_loss))
                if (ep % 1000 == 0) and (ep >0): #save files
                    # save parameters, save weights, save some test data, save model ckpt
                    epoch_path = modify_path(save_path)
                    model.save(epoch_path)
                    utils.save_parameters(opts, os.path.join(epoch_path,
                                                             opts.parameter_name))
                    model.save_weights(epoch_path)
                    model.save_activity(next_element[0], next_element[1],
                                        epoch_path)
                    with open(os.path.join(save_path, opts.log_name + '.pkl'),
                              'wb') as f:
                        pkl.dump(logger,f)

            #save latest
            model.save()
            utils.save_parameters(opts, os.path.join(save_path, opts.parameter_name))
            model.save_weights()
            model.save_activity(next_element[0], next_element[1])
            with open(os.path.join(save_path, opts.log_name + '.pkl'), 'wb') as f:
                pkl.dump(logger, f)

if __name__ == '__main__':
    st_model_opts = config.stationary_model_config()
    non_st_model_opts = config.non_stationary_model_config()
    opts = non_st_model_opts
    train(opts)






