import argparse
import config
import os
from train import train
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
    rnn_size = 100

    st_model_opts = config.stationary_model_config()
    st_model_opts.save_path = opts.stationary_path
    st_model_opts.rnn_size = rnn_size

    nonst_model_opts = config.non_stationary_model_config()
    nonst_model_opts.save_path = opts.nonstationary_path
    nonst_model_opts.dir_weights = os.path.join(st_model_opts.save_path, st_model_opts.file_name + '.pkl')
    nonst_model_opts.rnn_size = rnn_size
    nonst_model_opts.velocity_max = 3

    first = copy.deepcopy(st_model_opts)
    first.epoch = int(201)
    first.load_checkpoint= False

    first_more = copy.deepcopy(st_model_opts)
    first_more.epoch = int(201)
    first_more.load_checkpoint= True

    second = copy.deepcopy(nonst_model_opts)
    second.epoch = int(301)
    second.load_checkpoint = False
    second.load_weights = True
    second.time_steps = 25
    second.time_loss_end = 25
    second.velocity_use = [1]
    second.dir_weights = './curriculum/stationary/weight.pkl'

    second_more = copy.deepcopy(nonst_model_opts)
    second_more.epoch = int(301)
    second_more.load_checkpoint = True
    second_more.time_steps = 25
    second_more.time_loss_end = 25
    second_more.velocity_use = [1]
    
    second_last = copy.deepcopy(nonst_model_opts)
    second_last.epoch = int(1)
    second_last.load_checkpoint = True
    second_last.time_steps = 25
    second_last.time_loss_end = 25
    second_last.velocity_use = [1]
    second_last.batch_size= 40
    
    third = copy.deepcopy(nonst_model_opts)
    third.epoch = int(4e4 + 1)
    third.load_checkpoint = True
    third.velocity_use = [1,2]
    third.time_steps = 25
    third.time_loss_end = 25

    third_more = copy.deepcopy(nonst_model_opts)
    third_more.load_checkpoint = True
    third_more.epoch = int(2e4 + 1)
    third_more.velocity_use = [1,2]
    third_more.time_steps = 25
    third_more.time_loss_end = 25
    
    fourth = copy.deepcopy(nonst_model_opts)
    fourth.load_checkpoint = True
    fourth.epoch = int(5e4 + 1)
    fourth.velocity_use = [2]
    fourth.time_steps = 25
    fourth.time_loss_end = 25

    # c= [first, first_more, second]
    c = [second_more, second_last]
    return c

if __name__ == '__main__':
    #stationary
        curriculum = curriculum()
        for i, c in enumerate(curriculum):
            # with tf.Graph().as_default() as graph:
            #     X, Y = inputs.create_inputs(c)
            #     rnn = train.RNN(c)
            #     with tf.Session() as sess:
            #         print(c.__dict__)
            #         sess.run(tf.global_variables_initializer())
            #         sess.run(rnn.train_iter.initializer,
            #                  feed_dict={rnn.X_pl: X, rnn.Y_pl: Y})
            #         rnn.train(c)
            #         # e = c.test_batch_size
            #         # rnn.run_test(X[:e, :, :], Y[:e, :, :], c)

            train(c)
            print('[!] Curriculum %d has finished' % (i))
                    # time.sleep(2)
