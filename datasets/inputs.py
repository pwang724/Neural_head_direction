import numpy as np
from scipy.stats import norm
import config
from datasets.noise import ou_noise
import datasets.plot


def _nonneg(inputs):
    C = inputs[:, :, 0]
    S = inputs[:, :, 1]

    Cpos, Cneg, Spos, Sneg = np.zeros_like(C), np.zeros_like(C), np.zeros_like(C), np.zeros_like(C)
    Cpos[C >= 0] = C[C >= 0]
    Cneg[C < 0] = np.abs(C[C < 0])
    Spos[S >= 0] = S[S >= 0]
    Sneg[S < 0] = np.abs(S[S < 0])
    B = np.stack([Cpos, Cneg, Spos, Sneg], axis=2)
    return B

def _convert(start_positions, opts, mode):
    def _make_bumps(ix, state_size, bump_size, bump_std):
        assert bump_size < state_size, "bump size cannot be bigger than state size"

        span = bump_size // 2
        if bump_size % 2:  # odd
            bumps = [i for i in range(-span, span + 1)]
            middle_bump = span
        else:  # even
            bumps = [i for i in range(-span, span + 1) if i != 0]
            middle_bump = span - 1
        bumps = _sum_normalize(norm.pdf(bumps, scale=bump_std))

        input = np.zeros((len(ix), state_size))
        input[:, :bump_size] += bumps
        input = np.stack([np.roll(ipt, s) for ipt, s in zip(input, ix - middle_bump)], axis=0)
        return input

    def _sum_normalize(x):
        """Normalize by sum row-wise."""
        if len(x.shape) > 1:  # matrix
            x /= np.sum(x, axis=1).reshape(-1, 1)
        else:  # vector
            x = x / np.sum(x)
        return x

    if mode == 'trig':
        start_positions = start_positions * np.pi * 2 / 360
        C, S = np.cos(start_positions).reshape(-1,1), np.sin(start_positions).reshape(-1,1)
        positions = np.concatenate((C, S), axis=1)
    elif mode == 'onehot':
        interval = 360 / opts.state_size
        start_positions = np.floor(start_positions/interval)
        positions = np.zeros((start_positions.shape[0], opts.state_size))
        positions[np.arange(start_positions.shape[0]), start_positions.astype(int)] = 1
    elif mode == 'bump':
        interval = 360 / opts.state_size
        start_positions /= interval
        positions = _make_bumps(start_positions.astype(int), opts.state_size, opts.bump_size, opts.bump_std)
    else:
        raise AssertionError('{} not recognized'.format(opts.input_mode))
    return positions

def create_inputs(opts):
    """
    Create inputs and labels for training.
    """
    assert opts.input_mode in ['bump', 'trig', 'onehot'], "Invalid output mode"
    assert opts.output_mode in ['bump', 'trig', 'onehot'], "Invalid output mode"

    start_positions = np.random.random_sample(size=opts.n_input) * 360  # degrees
    start_positions = np.mod(start_positions, 360)
    if not opts.discrete:
        assert opts.input_mode == 'trig' and opts.output_mode == 'trig', 'cannot discretize angles if mode is not trig'
    else:
        interval = 360 / opts.state_size
        mod = opts.velocity_max / interval
        assert np.equal(np.mod(mod, 1), 0), 'maximum velocity is not divisible by velocity increments: {}'.format(mod)
        start_positions = np.floor(start_positions/interval) * interval

    positions = _convert(start_positions, opts, opts.input_mode)
    X = np.zeros((opts.n_input, opts.time_steps, positions.shape[-1]))
    for i, x in enumerate(positions):
        X[i,0,:] = x

    if opts.non_negative_input and opts.input_mode == 'trig':
        X = _nonneg(X)

    if opts.noise:
        assert 0 <= opts.noise_density <= 1, "Density is not between 0 and 1"
        assert 0 <= opts.noise_intensity, "Intensity is less than 0"
        noise = np.random.normal(scale=opts.noise_intensity, size=(opts.n_input, opts.time_steps, X.shape[2]))
        noise_mask = np.random.uniform(size=noise.shape)
        noise[:, 0, :] = 0
        noise[noise_mask < opts.noise_density] *= 0
        X += noise

    if opts.velocity:
        vel_ix = np.arange(opts.velocity_start, opts.time_steps, opts.velocity_gap)
        vel_per_batch = len(vel_ix)
        vel_one_hot = np.zeros((opts.n_input, opts.time_steps, 2))
        positions = np.zeros((opts.n_input, opts.time_steps))

        for i in range(opts.n_input):
            velocity = ou_noise(n=vel_per_batch, sigma=opts.velocity_max, tau=5)
            if opts.discrete:
                velocity_interval = 360 / opts.state_size
                velocity = np.round(velocity / velocity_interval) * velocity_interval

            max_velocity = opts.velocity_max
            velocity[velocity > max_velocity] = max_velocity
            velocity[velocity < -max_velocity] = -max_velocity
            velocity_norm = velocity / max_velocity
            position = start_positions[i]
            positions[i, 0:] = position
            for j, ix in enumerate(vel_ix):
                if velocity_norm[j] > 0:
                    vel_one_hot[i, ix, 1] = velocity_norm[j]
                else:
                    vel_one_hot[i, ix, 0] = - velocity_norm[j]
                position += velocity[j]
                position = np.mod(position, 360)
                positions[i, ix:] = position

        list_of_labels = []
        for position in positions:
            labels = _convert(position, opts, opts.output_mode)
            list_of_labels.append(labels)

        X = np.concatenate([X, vel_one_hot], axis=2)
        Y = np.stack(list_of_labels, axis=0)
    else:
        vel_one_hot = np.zeros((opts.n_input, opts.time_steps, 2))
        X = np.concatenate([X, vel_one_hot], axis=2) #pad it for ease of transfer later
        Y = np.zeros((opts.n_input, opts.time_steps, positions.shape[-1]))
        for i, x in enumerate(positions):
            Y[i, :, :] = x

    if opts.non_negative_output and opts.output_mode == 'trig':
        Y = _nonneg(Y)

    # for i in range(3):
    #     plt.subplot(1,2,1)
    #     plt.imshow(X[i,:,:])
    #     plt.subplot(1,2,2)
    #     plt.imshow(Y[i,:,:])
    #     plt.colorbar()
    #     plt.show()

    return X.astype(np.float32), Y.astype(np.float32)


if __name__ == '__main__':
    stationary = config.stationary_input_config()
    non_stationary = config.non_stationary_input_config()

    opts = stationary
    opts = non_stationary

    opts.time_steps = 50
    opts.bump_size = 3
    opts.state_size = 36
    opts.velocity_gap = 3
    opts.velocity_max = 30

    opts.input_mode = 'onehot'
    opts.output_mode = 'onehot'

    opts.nonneg_input = False
    inputs, labels = create_inputs(opts)

    print(inputs.shape)
    print(labels.shape)

    datasets.plot.plot_stationary_inputs(inputs, labels, opts)
    # plot_moving_inputs(inputs, labels, opts)
    # plot_nonneg(inputs, labels, opts)
    # plot_scalar_inputs(inputs, labels, vel_ipt, non_stationary)

