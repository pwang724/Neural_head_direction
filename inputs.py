import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import config
from utils import adjust
import os

# set seed for reproducibility
# np.random.seed(10)
def sum_normalize(x):
    """Normalize by sum row-wise."""
    if len(x.shape) > 1:  # matrix
        x /= np.sum(x, axis=1).reshape(-1,1)
    else:  # vector
        x = x / np.sum(x)
    return x

def random_shift(x, size, correlated=True):
    shift = []
    low, high = 2, 4
    while len(shift) < size:
        if correlated:
            duration = np.random.randint(low=low, high=high)
        else:
            duration = 1
        value = np.random.choice(x)
        shift += [value] * duration

    shift = shift[:size]
    return np.array(shift)

def make_bumps(ix, opts):
    env_size = opts.state_size
    bump_size = opts.bump_size
    bump_std = opts.bump_std
    assert bump_size < env_size, "bump size cannot be bigger than state size"

    span = bump_size // 2
    if bump_size % 2:  # odd
        bumps = [i for i in range(-span, span + 1)]
        middle_bump = span
    else:  # even
        bumps = [i for i in range(-span, span + 1) if i != 0]
        middle_bump = span - 1
    bumps = sum_normalize(norm.pdf(bumps, scale=bump_std))

    input = np.zeros((len(ix), env_size))
    input[:, :bump_size] += bumps
    input = np.stack([np.roll(ipt, s) for ipt, s in zip(input, ix - middle_bump)], axis=0)
    return input

def create_labels(input, shift, ix, opts, lo=0, hi=None):
    """
    Create labels for a single starting state with no noise.
    :param input: 1D input
    :param shift: magnitude of shift, 1D vector
    :param ix: indices where each velocity command is applied, 1D vector
    :param time_steps: total trial length
    :return: labels
    """
    env_size = opts.state_size
    time_steps = opts.time_steps
    linear_track = opts.linear_track
    boundary_velocity = opts.boundary_velocity
    if hi is None:
        hi = env_size

    # take the first state, rotate it using shift
    labels = np.zeros((time_steps, env_size))
    labels += input
    cancel_ipt_ix = []
    if len(shift):
        cur = input
        for i, s in enumerate(shift):
            if linear_track:
                prev_posn = np.argmax(cur)
                new_posn = max(min(prev_posn + s, hi-1), lo)
                s = new_posn - prev_posn
                if s == 0 and not boundary_velocity:  # vel_ipt is invalid
                    cancel_ipt_ix.append(ix[i])

            cur = np.roll(cur, s)
            labels[ix[i]:, :] = cur
    return labels, cancel_ipt_ix

def create_trig_labels(labels, opts, lo=0, hi=None):
    env_size = opts.state_size
    rescale_env = opts.rescale_env
    if hi is None:
        hi = env_size

    rescale = np.pi * 2 / (hi-lo)
    scale = np.pi * 2 / env_size
    posn = np.argmax(labels, axis=1)
    if rescale_env:
        trig_posn = rescale * (posn-lo)
    else:
        trig_posn = scale * posn

    C, S = np.cos(trig_posn), np.sin(trig_posn)
    labels = np.stack([C,S], axis=1)
    return labels

def create_scalar_labels(labels, opts, lo=0, hi=None):
    """labels - (t x d) array of labels for a single trial"""
    env_size = opts.state_size
    rescale_env = opts.rescale_env
    if hi is None:
        hi = env_size
    posn = np.argmax(labels, axis=1)
    if rescale_env:
        posn -= lo
        sz = hi - lo
    else:
        sz = env_size

    if env_size <= 72:
        labels = 2 * posn / sz - 1
    else:
        labels = 4 * posn / sz - 2
    return labels[:, np.newaxis]  # add extra dimension for tensorflow


def convert_inputs(inputs, lo=0, hi=None):
    """Convert inputs to trigonometric inputs or scalar inputs"""
    env_size = opts.state_size
    velocity = opts.velocity
    input_mode = opts.input_mode
    rescale_env = opts.rescale_env
    startposn = inputs[:, 0, :]
    if hi is None:
        hi = env_size

    if velocity:
        startposn = startposn[:, :-2]
    posn = np.argmax(startposn, axis=1)

    if input_mode == 'trig':
        rescale = np.pi * 2 / (hi - lo)
        scale = np.pi * 2 / env_size
        if rescale_env:
            trig_posn = rescale * (posn - lo)
        else:
            trig_posn = scale * posn

        C, S = np.cos(trig_posn), np.sin(trig_posn)
        new_start = np.stack([C, S], axis=1)[:, np.newaxis,:]
        d2 = 2

    elif input_mode == 'scalar':
        if rescale_env:
            posn -= lo
            sz = hi-lo
        else:
            sz = env_size

        if env_size <= 72:
            start = 2 * posn / sz - 1
        else:
            start = 4 * posn-lo / sz - 2
        new_start = start[:, np.newaxis, np.newaxis]
        d2 = 1

    else:
        return inputs

    rest = np.zeros([inputs.shape[0], inputs.shape[1] - 1, d2])
    new_inputs = np.concatenate([new_start, rest], axis=1)
    if velocity:
        vel = inputs[:, :, -2:]
        new_inputs = np.concatenate([new_inputs, vel], axis=2)
    return new_inputs


def create_inputs(opts, training=True):
    """
    Create inputs and labels for training.
    """
    env_size = opts.state_size
    n_input = opts.n_input
    time_steps = opts.time_steps
    rnn_size = opts.rnn_size

    bump_size = opts.bump_size
    bump_std = opts.bump_std
    linear_track = opts.linear_track

    noise = opts.noise
    noise_density = opts.noise_density
    noise_intensity = opts.noise_intensity

    velocity = opts.velocity
    input_mode = opts.input_mode
    output_mode = opts.output_mode
    boundary_velocity = opts.boundary_velocity
    correlated_path = opts.correlated_path

    subtrack = opts.subtrack
    subtrack_minlen = opts.subtrack_minlen
    subtrack_maxlen = opts.subtrack_maxlen
    rescale_env = opts.rescale_env

    nav_output = opts.nav_output  # navigational output, direction and distance to a home spot
    home = opts.home
    if home in ['center', None]:
        home = int(env_size / 2)
    elif home == 'random':
        home = np.random.randint(env_size)

    pi = np.pi
    if subtrack:
        subtrack_len = np.random.randint(low=subtrack_minlen, high=subtrack_maxlen)
        print(f'\nsubtrack length: {subtrack_len}')
        lo = np.random.randint(env_size-subtrack_len)
        hi = lo + subtrack_len
        rescale = pi * 2 / (hi-lo)
        scale = np.pi * 2 / env_size
        print(f'low: {lo}, high: {hi}')
        print(f'trig rescaled, low: {0}, high: {(hi-lo) * rescale}')
        print(f'trig unscaled, low: {lo*scale}, high: {hi * scale}')
    else:
        subtrack_len = env_size
        lo = 0
        hi = env_size

    assert output_mode in ['bump', 'trig', 'scalar'], "Invalid output mode"

    def make_bumps(ix):
        assert bump_size < env_size, "bump size cannot be bigger than state size"

        span = bump_size // 2
        if bump_size % 2:  # odd
            bumps = [i for i in range(-span, span + 1)]
            middle_bump = span
        else:  # even
            bumps = [i for i in range(-span, span + 1) if i != 0]
            middle_bump = span - 1
        bumps = sum_normalize(norm.pdf(bumps, scale=bump_std))

        input = np.zeros((len(ix), env_size))
        input[:, :bump_size] += bumps
        input = np.stack([np.roll(ipt, s) for ipt, s in zip(input, ix - middle_bump)], axis=0)
        return input

    def create_labels(input, shift, ix, lo=0, hi=env_size):
        """
        Create labels for a single starting state with no noise.
        :param input: 1D input
        :param shift: magnitude of shift, 1D vector
        :param ix: indices where each velocity command is applied, 1D vector
        :param time_steps: total trial length
        :return: labels
        """
        # take the first state, rotate it using shift
        labels = np.zeros((time_steps, env_size))
        labels += input
        cancel_ipt_ix = []
        if len(shift):
            cur = input
            for i, s in enumerate(shift):
                if linear_track:
                    prev_posn = np.argmax(cur)
                    new_posn = max(min(prev_posn + s, hi-1), lo)
                    s = new_posn - prev_posn
                    if s == 0 and not boundary_velocity:  # vel_ipt is invalid
                        cancel_ipt_ix.append(ix[i])

                cur = np.roll(cur, s)
                labels[ix[i]:, :] = cur
        return labels, cancel_ipt_ix

    def create_trig_labels(labels):
        rescale = pi * 2 / (hi-lo)
        scale = np.pi * 2 / env_size
        posn = np.argmax(labels, axis=1)
        if rescale_env:
            trig_posn = rescale * (posn-lo)
        else:
            trig_posn = scale * posn

        C, S = np.cos(trig_posn), np.sin(trig_posn)
        labels = np.stack([C,S], axis=1)

        return labels

    def create_scalar_labels(labels):
        """labels - (t x d) array of labels for a single trial"""
        posn = np.argmax(labels, axis=1)
        if rescale_env:
            posn -= lo
            sz = hi - lo
        else:
            sz = env_size

        if env_size <= 72:
            labels = 2 * posn / sz - 1
        else:
            labels = 4 * posn / sz - 2
        return labels[:, np.newaxis]  # add extra dimension for tensorflow

    def convert_inputs(inputs):
        """Convert inputs to trigonometric inputs or scalar inputs"""
        startposn = inputs[:, 0, :]
        if velocity:
            startposn = startposn[:, :-2]
        posn = np.argmax(startposn, axis=1)

        if input_mode == 'trig':
            rescale = pi * 2 / (hi - lo)
            scale = np.pi * 2 / env_size
            if rescale_env:
                trig_posn = rescale * (posn - lo)
            else:
                trig_posn = scale * posn

            C, S = np.cos(trig_posn), np.sin(trig_posn)
            new_start = np.stack([C, S], axis=1)[:, np.newaxis,:]
            d2 = 2

        elif input_mode == 'scalar':
            if rescale_env:
                posn -= lo
                sz = hi-lo
            else:
                sz = env_size

            if env_size <= 72:
                start = 2 * posn / sz - 1
            else:
                start = 4 * posn-lo / sz - 2
            new_start = start[:, np.newaxis, np.newaxis]
            d2 = 1

        else:
            return inputs

        rest = np.zeros([inputs.shape[0], inputs.shape[1] - 1, d2])
        new_inputs = np.concatenate([new_start, rest], axis=1)
        if velocity:
            vel = inputs[:, :, -2:]
            new_inputs = np.concatenate([new_inputs, vel], axis=2)
        return new_inputs

    ix = np.random.randint(low=lo, high=hi, size=n_input)
    # ix = np.random.randint(low=0, high=state_size, size=n_input)
    # if training:
    # else:
    #     ix = np.arange(state_size)

    input_unpadded = make_bumps(ix)
    batch_size = len(ix)

    # pad the inputs for time. inputs are batch X time X STATE_SIZE
    input = np.expand_dims(input_unpadded, axis=1)
    pad = np.zeros((batch_size, time_steps - 1, env_size))
    input = np.concatenate((input, pad), axis=1)
    if velocity:
        velocity_start = opts.velocity_start
        velocity_gap = opts.velocity_gap
        velocity_min = opts.velocity_min
        velocity_max = opts.velocity_max
        velocity_size = opts.velocity_size
        velocity_step = opts.velocity_step
        assert velocity_start < time_steps, "first velocity command occurs after last time-step"

        one_hot_len = velocity_size
        vel_ix = np.arange(velocity_start, time_steps, velocity_gap)
        vel_per_batch = len(vel_ix)
        vel_total = vel_per_batch * batch_size
        vel_pos = np.arange(velocity_min, velocity_max+1, velocity_step)
        vel_options = np.hstack([-vel_pos, vel_pos])
        print(vel_options)

        # vel_shifts = correlated_random(vel_options, size=vel_total)
        vel_shifts = random_shift(vel_options, size=vel_total, correlated=correlated_path)
        vel_norm = vel_shifts / velocity_max
        vel_pos_onehot, vel_neg_onehot = vel_norm.copy(), vel_norm.copy()
        vel_pos_onehot[vel_shifts < 0] = 0  # positive movement is less than 0? okay
        vel_neg_onehot[vel_shifts > 0] = 0
        vel_neg_onehot = np.abs(vel_neg_onehot)
        vel_one_hot = np.stack((vel_pos_onehot, vel_neg_onehot), axis=1)
        vel_one_hot = vel_one_hot.reshape(batch_size, vel_per_batch, one_hot_len)

        vel_ipt = np.zeros((batch_size, time_steps, one_hot_len))
        vel_ipt[:, vel_ix, :] = vel_one_hot
        shift = vel_shifts.astype(int).reshape(batch_size, vel_per_batch)
        # create labels from input
        labels = [create_labels(x, s, vel_ix, lo, hi) for x, s in zip(input_unpadded, shift)]
        # labels = [create_labels(input_unpadded[2], shift[2], vel_ix)]
        cancel_ipt_ix = [lab[1] for lab in labels]
        labels = [lab[0] for lab in labels]
        for j, cancel in enumerate(cancel_ipt_ix):
            vel_ipt[j, cancel, :] = 0

        if opts.grid_input:  # speed cells and head direction at all times
            speed = np.sum(vel_ipt, axis=2)
            _direction = np.zeros_like(speed)
            _direction[vel_ipt[:,:,0] > 0] = -1
            _direction[vel_ipt[:,:,1] > 0] = 1
            direction = np.zeros_like(speed)
            # print(time_steps)
            for i, d in enumerate(_direction):
                for j in range(time_steps):
                    if d[j] != 0:
                        direction[i, j:] = d[j]
            vel_ipt = np.stack([speed, direction], axis=2)

        input = np.concatenate([input, vel_ipt], axis=2)
    else:
        labels = [create_labels(x, [], []) for x in input_unpadded]
        labels = [lab[0] for lab in labels]
        vel_ipt = None

    if output_mode == 'scalar':
        labels = [create_scalar_labels(lab) for lab in labels]
    elif output_mode == 'trig':
        labels = [create_trig_labels(lab) for lab in labels]
    else:
        pass  # for bump or linear_bump attractor, labels are done
    labels = np.stack(labels, axis=0)

    if noise:
        # sample noisy positions, sample noise for those positions, add noise to inputs
        assert 0 <= noise_density <= 1, "Density is not between 0 and 1"
        # assert 0 <= noise_intensity <= 1, "Intensity is not between 0 and 1"
        assert 0 <= noise_intensity, "Intensity is less than 0"

        noise = np.random.normal(scale=noise_intensity, size=(n_input, time_steps, rnn_size))
        noise_mask = np.random.uniform(size=noise.shape)
        noise[noise_mask < noise_density] *= 0  # take density % of noise
    else:
        noise = np.zeros((n_input, time_steps, rnn_size))

    if input_mode in ['scalar', 'trig']:
        input = convert_inputs(input)

    if nav_output:
        _home = home - np.argmax(labels, axis=2)
        dist_home = np.abs(_home) / np.amax(np.abs(_home))
        dir_home = np.sign(_home)
        nav_labels = np.stack([dist_home, dir_home], axis=2)
        labels = np.concatenate([labels, nav_labels], axis=2)

    if opts.nonneg_input:
        input, labels = nonneg(input, labels, opts)

    return input.astype(np.float32), labels.astype(np.float32), noise.astype(np.float32), vel_ipt

def nonneg_conv(A):
    C = A[:, :, 0]
    S = A[:, :, 1]

    Cpos, Cneg, Spos, Sneg = np.zeros_like(C), np.zeros_like(C), np.zeros_like(C), np.zeros_like(C)
    Cpos[C >= 0] = C[C >= 0]
    Cneg[C < 0] = np.abs(C[C < 0])
    Spos[S >= 0] = S[S >= 0]
    Sneg[S < 0] = np.abs(S[S < 0])
    B = np.stack([Cpos, Cneg, Spos, Sneg], axis=2)
    return B

def nonneg(inputs, labels, opts):
    if opts.input_mode == 'trig':
        inputs_ei = nonneg_conv(inputs)
        inputs = np.concatenate([inputs_ei, inputs[:,:,-2:]], axis=2)

    if opts.output_mode == 'trig':
        labels = nonneg_conv(labels)
    return inputs, labels


def plot_moving_inputs(inputs, labels, opts):
    rc = (2,3)
    state = [x[:, :opts.state_size] for x in inputs[:rc[0]]]
    extra = [x[:, opts.state_size:opts.state_size+2] for x in inputs[:rc[0]]]

    i = 0
    # for bump labels
    if opts.output_mode == 'bump':
        fig, ax = plt.subplots(rc[0], rc[1])
        labels = labels[:rc[0]]
        for batch in zip(state, extra, labels):
            for d in batch:
                plot_ix = np.unravel_index(i, rc)
                cur_ax = ax[plot_ix]
                adjust(cur_ax)
                plt.sca(cur_ax)
                plt.imshow(d, cmap='RdBu_r', vmin=-1, vmax=1)
                cb = plt.colorbar()
                cb.set_ticks([-1, 1])
                i += 1
    else:
        f1, ax1 = plt.subplots(2,1)
        rad = np.arctan2(labels[:,:,1], labels[:,:,0])
        plt.sca(ax1[0])
        plt.plot(np.mod(rad[0], 2*np.pi))
        plt.sca(ax1[1])
        plt.imshow(inputs[0,:,-2:].T)

        r, c = 10, 20
        f2, ax2 = plt.subplots(r, c)
        for i in range(r):
            for j in range(c):
                ax2[i,j].plot(np.mod(rad[i*10+j], 2 * np.pi))

    plt.show()

def plot_stationary_inputs(inputs, labels, opts):
    rc = (2, 2)
    fig, ax = plt.subplots(rc[0], rc[1])
    state = inputs[:rc[0]]
    labels = labels[:rc[0]]
    i = 0
    for batch in zip(state, labels):
        for d in batch:
            plot_ix = np.unravel_index(i, rc)
            cur_ax = ax[plot_ix]
            adjust(cur_ax)
            plt.sca(cur_ax)
            plt.imshow(d, cmap='RdBu_r', vmin=-.3, vmax=.3)
            cb = plt.colorbar()
            cb.set_ticks([0, .3])
            i += 1
    plt.show()


def plot_nonneg(inputs, labels, opts):
    f1, ax1 = plt.subplots(2, 1)
    C = labels[:, :, 0] + labels[:, :, 1]
    S = labels[:, :, 2] + labels[:, :, 3]
    rad = np.arctan2(C, S)
    plt.sca(ax1[0])
    plt.plot(np.mod(rad[0], 2 * np.pi))
    plt.sca(ax1[1])
    plt.imshow(inputs[0, :, -2:].T)
    plt.show()

def plot_scalar_inputs(inputs, labels, vel_ipt, opts):
    rc = (2, 3)
    state_size = opts.state_size
    fig, ax = plt.subplots(rc[0], rc[1])
    state = [x[:, :state_size] for x in inputs[:rc[0]]]
    extra = [x[:, state_size:] for x in inputs[:rc[0]]]
    # labels = labels[:rc[0]]

    i = 0
    for batch in zip(state, extra, labels):
        for d in batch:
            if len(d.shape) == 1:
                d = d[:, np.newaxis]
            plot_ix = np.unravel_index(i, rc)
            cur_ax = ax[plot_ix]
            adjust(cur_ax)
            plt.sca(cur_ax)
            plt.imshow(d, cmap='RdBu_r', vmin=-1, vmax=1)
            cb = plt.colorbar()
            cb.set_ticks([-1, 1])
            i += 1

    state_size = opts.state_size
    vel_active = np.sum(vel_ipt, axis=2) > 0
    if state_size <= 72:
        ylim = [-1,1]
    else:
        ylim = [-2, 2]

    f1, ax1 = plt.subplots(10, 10)
    r, c = 0, 0
    while r < 10:
        ax1[r, c].plot(labels[r*10 + c])
        ax1[r, c].set_ylim(ylim)
        # vel = np.squeeze(np.argwhere(vel_active[r*10 + c]))
        # for v in vel:
        #     ax1[r, c].plot([v,v], ylim, c='gray', linestyle='--')

        c += 1
        if c >= 10:
            r += 1
            c = 0
    plt.show()


if __name__ == '__main__':
    stationary = config.stationary_input_config()
    non_stationary = config.non_stationary_input_config()

    opts = stationary
    opts = non_stationary

    opts.time_steps = 50
    opts.bump_size = 1
    opts.state_size = 72
    opts.velocity_gap = 3
    opts.velocity_step = 1
    opts.velocity_max = 3

    opts.linear_track = False
    opts.input_mode = 'trig'
    opts.output_mode = 'trig'
    # opts.input_mode = 'bump'
    # opts.output_mode = 'bump'

    # opts.bio_input = False
    opts.subtrack = False
    opts.rescale_env = False

    opts.nonneg_input = False
    inputs, labels, vel_ipt, _ = create_inputs(opts)

    print(inputs.shape)
    print(labels.shape)

    b = np.argmax(inputs[0, 0, :])
    # plot_stationary_inputs(inputs,labels, stationary)
    plot_moving_inputs(inputs, labels, opts)
    # plot_nonneg(inputs, labels, opts)
    # plot_scalar_inputs(inputs, labels, vel_ipt, non_stationary)

