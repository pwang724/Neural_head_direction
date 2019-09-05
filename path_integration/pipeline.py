import numpy as np
from old import train
# import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
from config import Options
import utils


def prepare_opts():
    opts = Options()
    opts.n_examples = int(1e3)
    opts.n_epoch = int(5e2)
    opts.rnn_type = 'tanh'  # no gains from relu at a short distance
    opts.folder = opts.rnn_type
    opts.network_state_size = 100
    opts.max_angle_change = 90
    opts.batch_size = 50
    opts.label_units = 'rad'  # rad, deg - rad is imprecise, but deg is useless
    opts.direction = 'current'  # current, home  # home seems to work better, but it really should work both ways
    opts.separate_losses = False  # useless - don't do this
    opts.zero_start = False  # helps with 2-leg (obv)
    opts.angle_format = 'trig'
    opts.output_format = 'cartesian'
    # polar with gap can follow existence of turns, but is very imprecise
    # works well in cartesian
    opts.velocity = True  # either form works
    opts.stopping_prob = .3

    opts.network_loss = True
    if opts.network_loss:
        opts.name = opts.rnn_type + '_awloss'
    else:
        opts.name = opts.rnn_type + '_noloss'

    opts.step_gap = 3
    opts.use_gap = False  # use of gap improves performance for polar coords, not needed in cartesian
    if opts.use_gap:
        opts.name += '_gap'

    # opts.name += '_{}n'.format(opts.network_state_size)
    print(opts.name)

    return opts

def run_experiment(opts, stages=[False]*3):
    # do I need curriculum training for longer periods of time or larger angles?
    assert opts.direction in ['current', 'home'], "Invalid direction type"

    stages = [True, True, True, False]
    if stages[0]:
        print('First training')
        opts.training_stage = 1
        opts.max_steps_dir = 5
        opts.agent_steps = 20
        opts = train.run_training(opts)
        opts.save()

    if stages[1]:
        print('\nSecond training')
        opts.training_stage = 1
        opts.load()
        opts.max_steps_dir = 15
        opts.agent_steps = 40
        opts.training_stage = 2
        opts = train.run_training(opts)
        opts.save()

    if stages[2]:
        print('\nFinal training')
        opts.training_stage = 2
        opts.load()
        opts.max_steps_dir = opts.d_step
        opts.agent_steps = 80
        opts.training_stage = 3
        opts = train.run_training(opts)
        opts.save()

    if stages[3]:
        print('extra training')
        opts.training_stage = 3
        opts.load()
        opts.max_steps_dir = opts.d_step
        opts.agent_steps = 200
        opts.training_stage = 4
        opts = train.run_training(opts)
        opts.save()

    return opts

def analysis(opts):
    opts.load()
    opts.n_examples = opts.batch_size
    opts.test = True  # n_examples is set by possible angle directions
    opts.test_type = 'full'  # full, 2-leg
    opts.r0 = 10
    opts.r1 = 5
    opts.agent_steps = 100
    opts.max_angle_change = 45

    W_ah, W_sh, W_hh, W_out, bias = utils.load_weights(opts.folder, opts.weights_name)
    inputs, labels, pred, states, angle_trig, opts = train.run_test(opts)

    titles = ['W_ah', 'W_sh', 'W_hh', 'W_out']
    mat = [W_ah, W_sh, W_hh, W_out]
    tup = zip(titles, mat)
    fname = 'figures/' + opts.get_path() + '_unsorted'
    # utils.pretty_image(tup, 2, 2, fname, cbar=True)

    # in labels and predictions, data axis 2 has angles (radians) and distance
    ix = 14
    # print('labels', labels[ix])
    # print('pred', pred[ix])
    if opts.output_format == 'polar':
        visualize_polar_trajectory(pred, angle_trig, ix, opts)
        # visualize_trajectory(labels, angle_trig, ix, opts)
    else:
        visualize_cartesian_trajectory(pred, angle_trig, ix, opts)
    plt.show()

def gaps(A, gap):
    t = gap + 1
    if len(A.shape) < 3:
        A = A.reshape(A.shape[0], A.shape[1], 1)

    d0, d1, d2 = A.shape
    Apad = np.zeros([d0, d1 * (gap + 1), d2])
    for i in range(A.shape[1]):
        Apad[:, i*t:(i+1)*t, :] = np.tile(A[:, i:i+1, :], [1, t, 1])

    return Apad

def visualize_polar_trajectory(pred, trig, ix, opts):
    loc = np.cumsum(trig, axis=1)
    path = np.concatenate([np.zeros([pred.shape[0], 1, 2]), loc], axis=1)  # add zeros for the starting position
    if opts.use_gap:
        loc = gaps(loc, opts.step_gap)

    if opts.label_units == 'deg':
        pred[:,:,0] *= np.pi / 180

    pred_ang = pred[:, :, 0]
    pred_dist = pred[:, :, 1]
    direction = np.stack([np.cos(pred_ang), np.sin(pred_ang)], axis=2)

    ftrig, axtrig = plt.subplots()
    axtrig.plot(path[ix, :, 0], path[ix, :, 1], linewidth=2, color='k')
    if opts.direction == 'current':
        # plot predicted path
        v = [pred_dist * direction[:, :, i] for i in range(2)]
        v = np.stack(v, axis=2)
        v = np.concatenate([np.zeros([v.shape[0], 1, 2]), v], axis=1)
        axtrig.plot(v[ix, :, 0], v[ix, :, 1], linewidth=2, color='b')
        plt.title('Polar coordinate path')

    else:  # opts.direction == 'home':
        # make vector from current position to predicted start position
        v = [loc[:, :, i] + pred_dist * direction[:, :, i] for i in range(2)]
        v = np.stack(v, axis=2)
        label_vec = np.stack([loc[ix], v[ix]], axis=1)
        for vec in label_vec:
            axtrig.plot(vec[:, 0], vec[:, 1], linestyle='--')
        plt.title('Polar coordinate return')

    axtrig.axis('equal')

def visualize_cartesian_trajectory(pred, trig, ix, opts):
    # plot 20 paths
    path = np.cumsum(trig, axis=1)
    path = np.concatenate([np.zeros([pred.shape[0], 1, 2]), path], axis=1)
    pred = np.concatenate([np.zeros([pred.shape[0], 1, 2]), pred], axis=1)

    M = [(path[k], pred[k]) for k in range(20)]
    titles = [None] * 20
    tup = zip(titles, M)
    fname = 'figures/' + opts.get_path() + '_paths'
    utils.pretty_plot(tup, 5, 4, fname, hide_ticks=True, ax_op='equal', suptitle='N={}'.format(opts.agent_steps))

    # print(path[ix].shape)
    # f, ax = plt.subplots()
    # ax.plot(path[ix, :, 0], path[ix, :, 1], linewidth=2, color='k')
    # ax.plot(pred[ix, :, 0], pred[ix, :, 1], linewidth=2, color='b')
    # ax.axis('equal')
    # plt.title('Cartesian coordinate path')

def pipeline():
    opts = prepare_opts()
    opts = run_experiment(opts)
    opts.training_stage = 4
    analysis(opts)


if __name__ == '__main__':
    pipeline()
