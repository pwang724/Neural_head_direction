import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import config
import noise

# set seed for reproducibility
# np.random.seed(2)
def sum_normalize(x):
    """Normalize by sum row-wise."""
    if len(x.shape) > 1: #matrix
        x /= np.sum(x, axis=1).reshape(-1,1)
    else: # vector
        x = x / np.sum(x)
    return x

def correlated_random(x, size):
    l = []
    low, high = 2, 4
    while len(l) < size:
        duration = np.random.randint(low=4, high=8)
        value = np.random.choice(x)
        for i in range(duration):
            l.append(value)
    l = l[:size]
    return np.array(l)

def create_inputs(opts):
    """
    Create inputs and labels for training.
    """
    state_size = opts.state_size
    n_input = opts.n_input
    time_steps = opts.time_steps

    bump_size = opts.bump_size
    bump_std = opts.bump_std

    noise = opts.noise
    noise_density = opts.noise_density
    noise_intensity = opts.noise_intensity

    velocity = opts.velocity


    def make_bumps(ix):
        assert bump_size < state_size, "bump size cannot be bigger than state size"

        span = bump_size // 2
        if bump_size % 2:  # odd
            bumps = [i for i in range(-span, span + 1)]
            middle_bump = span
        else:  # even
            bumps = [i for i in range(-span, span + 1) if i != 0]
            middle_bump = span - 1
        bumps = sum_normalize(norm.pdf(bumps, scale=bump_std))

        input = np.zeros((len(ix), state_size))
        input[:, :bump_size] += bumps
        input = np.stack([np.roll(ipt, s) for ipt, s in zip(input, ix - middle_bump)], axis=0)
        return input

    def create_labels(input, shift, ix):
        """
        Create labels for a single starting state with no noise.
        :param input: 1D input
        :param shift: magnitude of shift, 1D vector
        :param ix: indices where each velocity command is applied, 1D vector
        :param time_steps: total trial length
        :return: labels
        """
        # take the first state, rotate it using shift
        labels = np.zeros((time_steps, state_size))
        labels += input
        if len(shift):
            cur = input
            for i, s in enumerate(shift):
                cur = np.roll(cur, s)
                labels[ix[i]:, :] = cur
        return labels


    ix = np.random.randint(low=0, high=state_size, size=n_input)
    input_unpadded = make_bumps(ix)
    batch_size = len(ix)

    # pad the inputs for time. inputs are batch X time X STATE_SIZE
    input = np.expand_dims(input_unpadded, axis=1)
    pad = np.zeros((batch_size, time_steps - 1, input.shape[2]))
    input = np.concatenate((input, pad), axis = 1)

    if velocity:
        velocity_start = opts.velocity_start
        velocity_gap = opts.velocity_gap
        velocity_use = opts.velocity_use
        velocity_max = opts.velocity_max
        assert velocity_start < time_steps, "first velocity command occurs after last time-step"

        one_hot_len = 2 * velocity_max
        vel_ix = np.arange(velocity_start, time_steps, velocity_gap)
        vel_per_batch = len(vel_ix)
        vel_total = vel_per_batch * batch_size
        vel_options = np.array([-1 * l for l in velocity_use] + [l-1 for l in velocity_use])

        vel_direction = correlated_random(vel_options + velocity_max, size= vel_total)
        vel_one_hot = np.zeros((vel_total, one_hot_len))
        vel_one_hot[np.arange(vel_total), vel_direction] = 1
        vel_one_hot = vel_one_hot.reshape(batch_size, vel_per_batch, one_hot_len)
        velocity = np.zeros((input.shape[0], input.shape[1], one_hot_len))
        velocity[:, vel_ix, :] = vel_one_hot
        input = np.concatenate([input, velocity], axis=2)

        vel_options = np.array([i for i in range(-velocity_max, velocity_max+1) if i != 0])
        shift = vel_options[vel_direction]
        shift = shift.reshape(batch_size, vel_per_batch)

        # create labels from input

        labels = [create_labels(x, s, vel_ix) for x, s in zip(input_unpadded, shift)]
        labels = np.stack(labels, axis=0)
    else:
        labels = [create_labels(x, [], []) for x in input_unpadded]
        labels = np.stack(labels, axis=0)

    if noise:
        # sample noisy positions, sample noise for those positions, add noise to inputs
        assert 0 <= noise_density <= 1, "Density is not between 0 and 1"
        assert 0 <= noise_intensity <= 1, "Intensity is not between 0 and 1"
        max_noise = np.amax(input_unpadded[0]) * noise_intensity
        noise = np.random.uniform(low=0, high=max_noise, size=input.shape)
        noise[:,0,:]=0 #noise at first time point is zero
        noise_mask = np.random.uniform(size=noise.shape)
        noise[noise_mask < noise_density] *= 0  # take density % of noise
        input += + noise

    return input.astype(np.float32), labels.astype(np.float32)


if __name__ == '__main__':
    stationary = config.stationary_input_config()
    non_stationary = config.non_stationary_input_config()

    opts = non_stationary
    inputs, labels = create_inputs(opts)

    print(inputs.shape)
    print(labels.shape)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    plt.axis('off')
    ax[0].set_title('Input')
    sns.heatmap(inputs[0], vmin =0, vmax = .5, cbar=False, ax = ax[0])
    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax[0].yaxis.set_major_formatter(ticker.ScalarFormatter())

    ax[1].set_title('Label')
    sns.heatmap(labels[0], vmin=0, vmax=.5, cbar=True, ax = ax[1])
    ax[1].yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax[1].yaxis.set_major_formatter(ticker.ScalarFormatter())

    plt.show()
