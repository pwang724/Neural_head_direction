import argparse
import config
import os, re
import copy
import numpy as np
import pickle as pkl
import analysis
from tf_train import train
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stationary_path', type=str,
                        default='./curriculum/stationary', help='stationary path')
    parser.add_argument('--nonstationary_path', type=str,
                        default='./curriculum/non_stationary', help='non-stationary path')
    return parser

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


def stationary_curriculum(params):
    rnn_size, state_size = params
    parser = arg_parser()
    opts = parser.parse_args()
    opts.rnn_size = rnn_size

    loss = 'full'
    fn = 'relu'  # [relu, tanh]

    st_model_opts = config.stationary_model_config()
    # st_model_opts.output_mode = output_mode
    st_model_opts.weight_alpha = .5
    st_model_opts.activity_alpha = .1
    st_model_opts.rnn_size = rnn_size
    st_model_opts.state_size = state_size
    st_model_opts.losses = loss

    st_model_opts.activation_fn = fn
    st_model_opts.mask = True

    # first = copy.deepcopy(st_model_opts)
    # # first.epoch = int(201)
    # first.epoch = int(1)
    # first.load_checkpoint = False
    #
    # first_more = copy.deepcopy(st_model_opts)
    # # first_more.epoch = int(101)
    # first_more.epoch = int(201)
    # first_more.learning_rate = 1e-4
    # first_more.load_checkpoint = True

    # c = [first]
    # c = [first_more]
    # c = [first, first_more]


def moving_curriculum(params):
    # quick settings
    def fast_hd_trig(opts):
        opts.input_mode = 'trig'
        opts.output_mode = 'trig'
        opts.grid_input = False
        opts.linear_track = False
        return opts

    def fast_hd_pretty(opts):
        opts.input_mode = 'trig'
        opts.output_mode = 'bump'
        opts.grid_input = False
        opts.linear_track = False
        return opts

    def fast_hd_bump(opts):
        opts.input_mode = 'bump'
        opts.output_mode = 'bump'
        opts.grid_input = False
        opts.linear_track = False
        return opts

    def fast_grid(opts):
        opts.input_mode = 'bump'
        opts.output_mode = 'bump'
        opts.grid_input = True
        opts.linear_track = True
        return opts

    rnn_size, state_size = params
    parser = arg_parser()
    opts = parser.parse_args()

    nonst_model_opts = config.non_stationary_model_config()
    nonst_model_opts.weight_alpha = 0
    nonst_model_opts.activity_alpha = 0
    nonst_model_opts.rnn_size = rnn_size
    nonst_model_opts.state_size = state_size
    nonst_model_opts.losses = 'full'
    nonst_model_opts.activation_fn = 'relu'  # [relu, tanh]
    nonst_model_opts.mask = True
    nonst_model_opts.correlated_path = True

    nonst_model_opts = get_model_ix(nonst_model_opts)

    second = copy.deepcopy(nonst_model_opts)
    second.epoch = int(150)
    # second.epoch = int(1)
    second.load_checkpoint = False
    second.load_weights = False
    second.time_steps = 30
    second.time_loss_start = 5
    second.time_loss_end = second.time_steps
    second.velocity_step = 1
    second.velocity_min = 1
    second.velocity_max = 3
    second.learning_rate = .001
    # second.debug_weights = True

    # n_env = 10
    # second.subtrack = False
    # second.rescale_env = True
    # second.n_env = n_env

    ### Grid cells ###
    # second.grid_input = False
    # second.bump_in_network = False
    # nonst_model_opts.boundary_velocity = False

    second.dropout = False
    second.dropout_rate = .4

    second.noise = False
    second.noise_intensity = .05
    second.noise_density = .5

    ### Excitatory-inhibitory ###
    EI = True
    if EI:
        # second.time_steps = 30
        second.nonneg_input = True
        second.EI_in = True
        second.EI_h = True
        second.EI_out = True
    second.prop_ex = .9

    #### ASSIGN TRAINING PARADIGM ####
    second = fast_hd_trig(second)
    second = fast_hd_pretty(second)
    # second = fast_hd_bump(second)
    # second = fast_grid(second)

    second_more = copy.deepcopy(second)
    second_more.epoch = int(50)
    second_more.time_steps = 60
    second_more.time_loss_end = second_more.time_steps
    second_more.learning_rate = 1e-4
    second_more.load_checkpoint = True

    # c = [second]
    # c = [second_more]
    c = [second, second_more]
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
    c, _ = train(op)
    train_path = os.path.join(op.save_path, 'training_set.pkl')
    if use_data:
        with open(train_path, 'rb') as f:
            data_dict = pkl.load(f)
    else:
        data_dict = None
    analysis.plot_activity(c, data=data_dict)
    analysis.analyze_nonstationary_weights(c, plot=True, eval=False, load_df=False, data=data_dict)


if __name__ == '__main__':
    # rnn_sizes = [25, 36, 50, 64, 100]
    rnn_sizes = [100]
    params = [(s, 36) for s in rnn_sizes]  # rnn size, input size
    for p in params:
        cur = moving_curriculum(p)
        if cur[0].subtrack:
            subtrack_train(cur)
        else:
            for i, c in enumerate(cur):
                c, _ = train(c, seed=False)
                # c, _ = train(c, seed=True)
                print('[!] Curriculum %d has finished' % (i))
            # analysis.plot_activity(c)
            # analysis.analyze_nonstationary_weights(c, plot=True, eval=False, load_df=False)

