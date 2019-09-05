import os, re
import copy
import numpy as np
import pickle as pkl
import analysis
from old import tf_train
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
    nonst_model_opts = config.non_stationary_model_config()
    nonst_model_opts.weight_alpha = 0
    nonst_model_opts.activity_alpha = 0
    nonst_model_opts.losses = 'full'
    nonst_model_opts.activation_fn = 'relu'  # [relu, tanh]
    nonst_model_opts.mask = True
    nonst_model_opts.correlated_path = True

    nonst_model_opts = get_model_ix(nonst_model_opts)

    second = copy.deepcopy(nonst_model_opts)
    second.epoch = int(150)
    second.load_checkpoint = False
    second.load_weights = False
    second.time_steps = 30
    second.time_loss_start = 5
    second.time_loss_end = second.time_steps
    second.velocity_step = 1
    second.velocity_min = 1
    second.velocity_max = 3
    second.learning_rate = .001

    second.input_mode = 'trig' #or bump
    second.output_mode = 'trig' #or bump
    second.grid_input = False
    second.linear_track = False

    second.EI = True
    if second.EI:
        second.nonneg_input = True
        second.EI_in = True
        second.EI_h = True
        second.EI_out = True
        second.prop_ex = .9

    second.dropout = False
    second.dropout_rate = .4

    second.noise = False
    second.noise_intensity = .05
    second.noise_density = .5

    # second.debug_weights = True

    second_more = copy.deepcopy(second)
    second_more.epoch = int(50)
    second_more.time_steps = 60
    second_more.time_loss_end = second_more.time_steps
    second_more.learning_rate = 1e-4
    second_more.load_checkpoint = True

    # n_env = 10
    # second.subtrack = False
    # second.rescale_env = True
    # second.n_env = n_env

    ### Grid cells ###
    # second.grid_input = False
    # second.bump_in_network = False
    # nonst_model_opts.boundary_velocity = False

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
    c, _ = tf_train.train(op)
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
    if configs[0].subtrack:
        subtrack_train(configs)
    else:
        for i, c in enumerate(configs):
            c, _ = tf_train.train(c, seed=False)
            print('[!] Curriculum %d has finished' % (i))

        # analysis.plot_activity(c)
        # analysis.analyze_nonstationary_weights(c, plot=True, eval=False, load_df=False)

