import argparse
import config
import os
import train
import tensorflow as tf
import inputs
import copy
import time

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stationary_path', type=str,
                        default='./curriculum/stationary', help='stationary path')
    parser.add_argument('--nonstationary_path', type=str,
                        default='./curriculum/non_stationary', help='non-stationary path')
    return parser

def curriculum():
    parser = arg_parser()
    opts = parser.parse_args()
    rnn_size = 50

    st_model_opts = config.stationary_model_config()
    st_model_opts.save_path = opts.stationary_path
    st_model_opts.rnn_size = rnn_size

    nonst_model_opts = config.non_stationary_model_config()
    nonst_model_opts.save_path = opts.nonstationary_path
    pretrained_weight_dir = os.path.join(st_model_opts.save_path, st_model_opts.file_name + '.pkl')
    nonst_model_opts.dir_weights = pretrained_weight_dir
    nonst_model_opts.rnn_size = rnn_size

    first = copy.deepcopy(st_model_opts)
    first.epoch = int(2e4+1)
    first.load_checkpoint= False

    first_more = copy.deepcopy(st_model_opts)
    first_more.epoch = int(2e4+1)
    first_more.load_checkpoint= True

    second = copy.deepcopy(nonst_model_opts)
    second.epoch = int(2e4 +1)
    second.load_checkpoint = False
    second.load_weights = True
    second.time_steps = 25
    second.time_loss_end = 25

    second_more = copy.deepcopy(nonst_model_opts)
    second_more.epoch = int(2e4 + 1)
    second_more.load_checkpoint = True
    second_more.time_steps = 25
    second_more.time_loss_end = 25

    third = copy.deepcopy(nonst_model_opts)
    third.epoch = int(2e4 + 1)
    third.time_steps = 50
    third.time_loss_end = 50
    
    fourth = copy.deepcopy(nonst_model_opts)
    fourth.epoch = int(1e4 + 1)
    fourth.time_steps = 50
    fourth.time_loss_end = 50
    return [first, first_more, second, second_more, third]

if __name__ == '__main__':
    #stationary
        curriculum = curriculum()
        for i, c in enumerate(curriculum):
            with tf.Graph().as_default() as graph:
                with tf.Session() as sess:
                    print(c.__dict__)
                    rnn = train.RNN(sess, c)
                    sess.run(tf.global_variables_initializer())
                    X, Y = inputs.create_inputs(c)
                    rnn.run_training(X, Y, c)

                    print('[!] Curriculum %d has finished' % (i))
                    # time.sleep(2)
