import numpy as np
from config import *
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns


def sum_normalize(x):
    """Normalize by sum row-wise."""
    if len(x.shape) > 1:  # matrix
        x /= np.sum(x, axis=1).reshape(-1,1)
    else:  # vector
        x = x / np.sum(x)
    return x


def make_inputs(opts, plot=False):
    """Make the angle and step counter inputs for the ant path integration task."""
    # generate angles - angle indicates absolute direction of movement
    # move for a few steps in one direction, change
    state_size = opts.d_angle
    max_angle_change = opts.max_angle_change
    max_steps_dir = opts.max_steps_dir
    # 2-leg test
    r0 = opts.r0
    r1 = opts.r1
    velocity = opts.velocity
    if velocity:
        d_step = 1
        opts.d_step = 1
    else:
        d_step = max(opts.d_step, r0)

    bump_size = opts.bump_size
    bump_std = opts.bump_std
    angle_format = opts.angle_format
    test = opts.test
    test_type = opts.test_type
    gap = opts.step_gap

    pi = np.pi
    deg2rad = pi / 180
    scale = 2 * pi / state_size
    assert opts.label_units in ['rad', 'deg'], "Invalid angle units"

    # set of possible turns - calculate in degrees, convert to radians later
    angle_set = np.arange(state_size) * 360 / state_size
    _delta = angle_set[angle_set <= max_angle_change]
    delta = np.concatenate([_delta[1:], -_delta[1:]])
    _full_delta = angle_set[angle_set <= 180]
    full_delta = np.concatenate([_full_delta[0:], -_full_delta[1:]])

    T = opts.agent_steps
    N = opts.n_examples
    if test and test_type == '2-leg':
        T = r0 + r1
        N = full_delta.shape[0]

    def _step_inputs(steps):
        ipt = np.zeros([steps, d_step])
        for i in range(steps):
            if velocity:
                # move = np.zeros(d_step)
                # move[0] = 1
                # ipt[i] = move
                ipt[i] += 1
            else:
                ipt[i, :i+1] = 1
        return ipt

    def trig(A):
        """Create the inputs in the form of an angle in radians."""
        angle_trig = np.stack([np.cos(A), np.sin(A)], axis=2)
        # angle_input = np.arctan2(angle_trig[:,:,1], angle_trig[:,:,0])
        angle_input = np.mod(np.arctan2(angle_trig[:,:,1], angle_trig[:,:,0]) / deg2rad, 360) * deg2rad
        angle_input = angle_input[:,:, np.newaxis]

        trigsum = np.cumsum(angle_trig, axis=1)
        angle_labels = np.arctan2(trigsum[:,:,1], trigsum[:,:,0])
        angle_labels = np.mod(angle_labels / deg2rad, 360) * deg2rad
        # labels are the angle back to start - flip by 180 degrees...?
        if opts.direction == 'home':
            angle_labels = np.mod(angle_labels - pi, 2 * pi)

        com = np.mod(np.round(angle_labels / scale), state_size)
        return angle_trig, angle_input, angle_labels, com.astype(np.int32)

    def onehot(A):
        """Inputs as a one-hot vector of direction."""
        angle_trig, angle_input, angle_labels, com = trig(A)
        posn = np.round(A * state_size / (2*pi)).ravel().astype(np.int32)  # get vector position from angle
        angle_OH = np.zeros([N * T, state_size])
        angle_OH[np.arange(N*T), posn] = 1
        angle_OH = angle_OH.reshape(N, T, state_size)
        return angle_OH, angle_labels

    def make_bumps(ix):
        assert bump_size < state_size, "bump size cannot be bigger than state size"

        span = bump_size // 2
        if bump_size % 2:  # odd
            bumps = list(range(-span, span + 1))
            middle_bump = span
        else:  # even
            bumps = [i for i in range(-span, span + 1) if i != 0]
            middle_bump = span - 1
        bumps = sum_normalize(norm.pdf(bumps, scale=bump_std))

        input = np.zeros((len(ix), state_size))
        input[:, :bump_size] += bumps
        input = np.stack([np.roll(ipt, s) for ipt, s in zip(input, ix - middle_bump)], axis=0)
        return input

    def distribution(A):
        """Inputs as a distribution centered at the current direction."""
        angle_trig, angle_input, angle_labels, com = trig(A)

        posn = np.round(A * state_size / (2 * pi)).ravel().astype(np.int32)
        angle_dist = make_bumps(posn)
        angle_dist = angle_dist.reshape(N, T, state_size)
        return angle_dist, angle_labels

    def generate_paths():
        A = np.zeros([N, T])
        S = np.zeros([N, T, d_step])
        stop_trials = []
        for i in range(N):
            d = np.random.randint(1, max_steps_dir + 1)
            if not opts.zero_start:
                A[i, :d] = np.random.choice(angle_set)  # random start
            S[i, :d, :] = _step_inputs(d)
            while d < T:
                rem = T - d
                steps = min(np.random.randint(1, max_steps_dir + 1), rem)
                stop = np.random.rand() < opts.stopping_prob
                if stop:
                    turn = 0  # step signal is already blank
                    stop_trials.append(i)
                else:
                    turn = np.random.choice(delta)
                    S[i, d:d + steps] = _step_inputs(steps)

                A[i, d:d + steps] = A[i, d - 1] + turn
                d += steps

        return A, S

    def gaps(M, vel=False):
        # M is a (n x t x d matrix - extend to (n x (gap+1)t x d)
        t = gap + 1
        if len(M.shape) < 3:
            M = M.reshape(M.shape[0], M.shape[1], 1)

        d0, d1, d2 = M.shape
        Mpad = np.zeros([d0, d1 * (gap + 1), d2])
        if vel:  # if velocity on, only give on signal at movement step
            # move = np.zeros(d_step)
            # move[0] = 1
            # Mpad[:, ::t, :] = move
            Mpad[:, ::t, :] += 1
        else:
            for i in range(M.shape[1]):
                Mpad[:, i*t:(i+1)*t, :] = np.tile(M[:, i:i+1, :], [1, t, 1])

        return Mpad

    if test:
        angles = np.zeros([N, T])
        step_inputs = np.zeros([N, T, d_step])
        assert test_type in ['full', '2-leg'], 'Invalid test type'
        if test_type == 'full':
            np.random.seed(2)
            angles, step_inputs = generate_paths()
        else:
            # angles[:, r0:r0+r1] = np.stack([angle_set] * r1, axis=1)
            angles[:, :r0] += 90
            angles[:, r0:r0+r1] = np.stack([full_delta] * r1, axis=1)
            step_inputs[:, :r0] = np.tile(_step_inputs(r0), [N, 1]).reshape(N, r0, d_step)
            step_inputs[:, r0:r0+r1] = np.tile(_step_inputs(r1), [N, 1]).reshape(N, r1, d_step)
    else:
        angles, step_inputs = generate_paths()

    angles = np.mod(angles, 360) * deg2rad
    angle_trig, angle_input, angle_labels, _ = trig(angles)
    if angle_format == 'trig':
        pass
    elif angle_format == 'onehot':
        angle_input, labels = onehot(angles)
    elif angle_format == 'dist':
        angle_input, labels = distribution(angles)

    path = np.cumsum(angle_trig, axis=1)  # x-y coordinates
    step_labels = np.linalg.norm(path, axis=2)  # step labels are distance to start point
    if opts.label_units == 'deg':
        angle_labels /= deg2rad

    assert opts.output_format in ['polar', 'cartesian'], "Invalid output format"
    if opts.output_format == 'polar':
        labels = np.stack([angle_labels, step_labels], axis=2)
    else:
        labels = path

    if opts.use_gap:
        angle_input = gaps(angle_input)
        labels = gaps(labels)
        step_inputs = gaps(step_inputs, vel=velocity)

    # visualize
    if plot:
        ix = 9

        # plot inputs of each type
        _, A_trig, _, _ = trig(angles)
        A_OH, _ = onehot(angles)
        A_dist, _ = distribution(angles)

        foh, axoh = plt.subplots(4, 1)
        plt.sca(axoh[0])
        plt.scatter(np.arange(A_trig[ix].shape[0]), A_trig[ix])

        plt.sca(axoh[1])
        sns.heatmap(A_OH[ix])

        plt.sca(axoh[2])
        sns.heatmap(A_dist[ix])

        plt.sca(axoh[3])
        sns.heatmap(step_inputs[ix])

        # plot angle labels
        direction = np.stack([np.cos(angle_labels), np.sin(angle_labels)], axis=2)
        if opts.direction == 'current':
            # vector from start to predicted current position
            v = [step_labels * direction[:, :, i] for i in range(2)]
            v = np.stack(v, axis=2)
            label_vec = np.stack([np.zeros_like(v[ix]), v[ix]], axis=1)
        else:  # opts.direction == 'home':
            # vector from current position to predicted start position
            v = [path[:, :, i] + step_labels * direction[:, :, i] for i in range(2)]
            v = np.stack(v, axis=2)
            label_vec = np.stack([path[ix], v[ix]], axis=1)

        ftrig, axtrig = plt.subplots()
        axtrig.plot(path[ix,:,0], path[ix,:,1])
        for i in range(T):
            axtrig.plot(label_vec[i,:,0], label_vec[i,:,1], linestyle='--')
        axtrig.axis('equal')
        plt.show()

    inputs = np.concatenate([angle_input, step_inputs], axis=2)
    return inputs, labels, angle_trig, opts


if __name__ == '__main__':
    np.random.seed(2)
    opts = Options()
    # opts.angle_format = 'trig'
    opts.use_gap = True
    # opts.velocity = True
    opts.output_format = 'cartesian'
    opts.direction = 'current'
    make_inputs(opts, plot=True)

