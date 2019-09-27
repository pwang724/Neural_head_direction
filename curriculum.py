import matplotlib as mpl
mpl.use('TkAgg')

import os, re
import copy
import numpy as np
import pickle as pkl
import analysis
import torch_train as train
import config

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_model_ix(opts):
    modeldir = os.path.join(os.getcwd(), 'training')
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    exp = re.compile('model([0-9]+)')
    used_ix = []
    with os.scandir(modeldir) as models:
        for m in models:
            rg = exp.match(m.name)
            if rg:
                used_ix.append(int(rg.group(1)))

    if used_ix:
        max_ix = np.amax(used_ix)
        full_ix = np.arange(max_ix + 2)  # include the next index above the highest
        free_ix = [a for a in full_ix if a not in used_ix]
        new_ix = np.amin(free_ix)
    else:
        new_ix = 0
    opts.save_path = os.path.join(os.getcwd(), 'training', f'model{str(new_ix).zfill(3)}')
    opts.dir_weights = os.path.join(opts.save_path, opts.model_name + '.pkl')
    return opts


def stationary_curriculum():
    st_model_opts = config.stationary_model_config()
    st_model_opts.weight_alpha = .5
    st_model_opts.activity_alpha = .1
    st_model_opts.losses = 'full'
    st_model_opts.activation_fn = 'relu'
    st_model_opts.mask = True

    # first = copy.deepcopy(st_model_opts)
    # # first.epoch = int(201)
    # first.epoch = int(1)
    # first.load_checkpoint = False

    # c = [first]
    # c = [first_more]
    # c = [first, first_more]


def moving_curriculum():
    nonst_model_opts = config.modelConfig()
    nonst_model_opts.weight_alpha = .1
    nonst_model_opts.activity_alpha = .1

    nonst_model_opts = get_model_ix(nonst_model_opts)

    second = copy.deepcopy(nonst_model_opts)
    second.epoch = int(200)
    second.reload = False
    second.time_steps = 50
    second.time_loss_start = 5
    second.time_loss_end = second.time_steps
    second.learning_rate = .001

    # second.velocity_max = 30  # degrees
    # second.velocity_start = 5
    second.velocity_gap = 1  # every step has a command
    second.stop_probability = .2  # how often to stop moving
    second.velocity_onehot = False

    second.discrete = False
    second.input_mode = 'trig'  # trig, bump, onehot
    second.output_mode = 'trig'  # trig, bump, onehot
    second.grid_input = False
    second.non_negative_input = False
    second.non_negative_output = False

    second.constrained = False

    second.EI = False
    second.prop_ex = .8

    second.dropout = False
    second.dropout_rate = .2

    second.noise = False
    second.noise_intensity = .05
    second.noise_density = .5

    second.save_path = './_DATA/collaborator_settings'
    # second.save_path = './_DATA/' + second.input_mode + '_' + second.output_mode
    second.print_epoch_interval = 5
    second.save_epoch_interval = 100

    second.rng_seed = 0

    second_more = copy.deepcopy(second)
    second_more.epoch = int(200)
    second_more.time_steps = 50
    second_more.time_loss_end = second_more.time_steps
    second_more.learning_rate = 1e-4
    second_more.reload = True

    # n_env = 10
    # second.subtrack = False
    # second.rescale_env = True
    # second.n_env = n_env

    ### Grid cells ###
    # second.grid_input = False
    # second.bump_in_network = False
    # nonst_model_opts.boundary_velocity = False

    c = [second]
    # c = [second_more]
    # c = [second, second_more]
    return c


def subtrack_train(cur):
    """Training schedule for subtrack training. Run n_env times, then one more time with the full track."""
    assert cur[0].state_size >= cur[0].subtrack_maxlen, "Track length is longer than input size"
    op = cur[0]
    op.image_folder = 'image0'
    print("Round 0")
    run_track(op)

    op = cur[1]
    for env in range(op.n_env - 1):
        op.image_folder = f'image{env+1}'
        print(f"\nRound {env+1}")
        run_track(op)
    op.subtrack = False
    op.epoch = 50
    op.image_folder = 'imagefull'
    run_track(op)


def run_track(op, use_data=True):
    c, _ = train.train(op)
    train_path = os.path.join(op.save_path, 'training_set.pkl')
    if use_data:
        with open(train_path, 'rb') as f:
            data_dict = pkl.load(f)
    else:
        data_dict = None
    analysis.plot_activity(c, data=data_dict)
    analysis.analyze_nonstationary_weights(c, plot=True, eval=False, load_df=False, data=data_dict)


if __name__ == '__main__':
    configs = moving_curriculum()
    for i, c in enumerate(configs):
        train.train(modelConfig=c, reload=c.reload, set_seed=True)
        print('[!] Curriculum %d has finished' % (i))
    train.evaluate(modelConfig=c, log=True)

    # analysis.plot_activity(c)
    # analysis.analyze_nonstationary_weights(c, plot=True, eval=False, load_df=False)

