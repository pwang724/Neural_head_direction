import numpy as np
from config import *
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
# np.random.seed(2)


def sum_normalize(x):
    """Normalize by sum row-wise."""
    if len(x.shape) > 1: # matrix
        x /= np.sum(x, axis=1).reshape(-1,1)
    else:  # vector
        x = x / np.sum(x)
    return x


def make_inputs(opts, plot=False):
    """Make the angle and step counter inputs for the ant path integration task."""
    # generate angles - angle indicates absolute direction of movement
    # move for a few steps in one direction, change
    state_size = opts.state_size
    max_angle_change = opts.max_angle_change
    max_steps_dir = opts.max_steps_dir
    T = opts.timesteps
    N = opts.n_examples
    bump_size = opts.bump_size
    bump_std = opts.bump_std
    angle_format = opts.angle_format

    pi = np.pi
    deg2rad = pi / 180
    scale = 2 * pi / state_size

    # set of possible turns
    angle_set = np.arange(state_size) * 2 * pi / state_size
    _delta = angle_set[angle_set < (max_angle_change * deg2rad)]
    delta = np.concatenate([_delta[1:], -_delta[1:]])

    def _step_inputs(steps):
        ipt = np.zeros([steps, max_steps_dir])
        for i in range(steps):
            ipt[i, :i+1] = 1
        return ipt

    def trig(A):
        """Create the inputs in the form of an angle in radians."""
        angle_trig = np.stack([np.cos(A), np.sin(A)], axis=2)
        angle_input = np.arctan2(angle_trig[:,:,1], angle_trig[:,:,0])
        trigsum = np.cumsum(angle_trig, axis=1)
        angle_labels = np.arctan2(trigsum[:,:,1], trigsum[:,:,0])
        # labels are the angle back to start - flip by 180 degrees
        angle_labels = np.mod(angle_labels - pi, 2 * pi)
        com = np.round(angle_labels / scale) % state_size
        return angle_trig, angle_input, angle_labels, com.astype(np.int32)

    def onehot(A):
        """Inputs as a one-hot vector of direction."""
        angle_trig, angle_input, angle_labels, com = trig(A)

        posn = np.round(A * state_size / (2 * pi)).ravel().astype(np.int32)  # get vector position from angle
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

    # trials have variable steps in each direction and turns - generate each separately
    angles = np.zeros([N, T])
    step_inputs = np.zeros([N, T, max_steps_dir])
    # choose a starting direction
    for i in range(N):
        d = np.random.randint(1, max_steps_dir + 1)
        angles[i, :d] = np.random.choice(angle_set)
        step_inputs[i, :d] = _step_inputs(d)
        while d < T:
            rem = T - d
            steps = min(np.random.randint(1, max_steps_dir + 1), rem)
            turn = np.random.choice(delta)
            angles[i, d:d + steps] = angles[i, d - 1] + turn
            step_inputs[i, d:d + steps] = _step_inputs(steps)
            d += steps

    angles = np.mod(angles, 2 * pi)

    angle_trig, angle_input, angle_labels, _ = trig(angles)
    label_deg = angle_labels / deg2rad
    if angle_format == 'trig':
        pass
    elif angle_format == 'onehot':
        angle_input, labels = onehot(angles)
    elif angle_format == 'dist':
        angle_input, labels = distribution(angles)

    path = np.cumsum(angle_trig, axis=1)
    step_labels = np.linalg.norm(path, axis=2)  # step labels are distance to start point
    direction = np.stack([np.cos(angle_labels), np.sin(angle_labels)], axis=2)

    # visualize
    if plot:
        ix = 5

        # end position of walk from path to predicted start position
        v = [path[:, :, i] + step_labels * direction[:, :, i] for i in range(direction.shape[2])]
        v = np.stack(v, axis=2)

        angle_OH, labels = onehot(angles)
        foh, axoh = plt.subplots(3, 1)
        plt.sca(axoh[0])
        sns.heatmap(angle_OH[ix])

        angle_dist, labels = distribution(angles)
        plt.sca(axoh[1])
        sns.heatmap(angle_dist[ix])

        plt.sca(axoh[2])
        sns.heatmap(step_inputs[ix])

        ftrig, axtrig = plt.subplots()
        axtrig.plot(path[ix,:,0], path[ix,:,1])
        label_vec = np.stack([path[ix, :, :], v[ix, :, :]], axis=1)
        for i in range(T):
            axtrig.plot(label_vec[i,:,0], label_vec[i,:,1], linestyle='--')
        axtrig.axis('equal')
        plt.show()

    return angle_input, step_inputs, angle_labels, step_labels, angle_trig


if __name__ == '__main__':
    opts = Options()
    make_inputs(opts, plot=True)

