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

def curriculum(rnn_size):
    parser = arg_parser()
    opts = parser.parse_args()

    st_model_opts = config.stationary_model_config()
    st_model_opts.save_path = os.path.join(os.getcwd(), 'lab_meeting',
                                           str(rnn_size).zfill(3), 'stationary')
    st_model_opts.rnn_size = rnn_size
    st_model_opts.weight_alpha = 1
    st_model_opts.activity_alpha = .1

    nonst_model_opts = config.non_stationary_model_config()
    nonst_model_opts.save_path = os.path.join(os.getcwd(), 'lab_meeting',
                                           str(rnn_size).zfill(3),
                                              'non_stationary')
    nonst_model_opts.dir_weights = os.path.join(st_model_opts.save_path, st_model_opts.file_name + '.pkl')
    nonst_model_opts.rnn_size = rnn_size
    nonst_model_opts.velocity_max = 1
    nonst_model_opts.weight_alpha = 1
    nonst_model_opts.activity_alpha = .1

    first = copy.deepcopy(st_model_opts)
    first.epoch = int(201)
    first.load_checkpoint= False

    first_more = copy.deepcopy(st_model_opts)
    first_more.epoch = int(301)
    first_more.load_checkpoint= True

    second = copy.deepcopy(nonst_model_opts)
    second.epoch = int(1)
    second.load_checkpoint = False
    second.load_weights = True
    second.time_steps = 20
    second.time_loss_start = 5
    second.time_loss_end = 20
    second.velocity_use = [1]
    second.dir_weights = os.path.join(st_model_opts.save_path,
                                      st_model_opts.weight_name + '.pkl')

    second_more = copy.deepcopy(nonst_model_opts)
    second_more.epoch = int(201)
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
    # second_last.velocity_max = 3
    # second_last.velocity_size = 6
    # second_last.save_path = './gold_copy/non_stationary'
    # c= [first, first_more]
    c = [second]
    return c

if __name__ == '__main__':
        curriculum = curriculum(70)
        for i, c in enumerate(curriculum):
            train(c)
            print('[!] Curriculum %d has finished' % (i))
