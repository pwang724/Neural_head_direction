import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# set seed for reproducibility
# np.random.seed(2)

def sum_normalize(x):
    """Normalize by sum row-wise."""
    if len(x.shape) > 1: #matrix
        x /= np.sum(x, axis=1).reshape(-1,1)
    else: # vector
        x = x / np.sum(x)
    return x

def create_inputs(n_samples, state_size, time_steps=5, bump_size=1, bump_std=1,
                  noise=False, noise_intensity=.5, noise_density=.5,
                  velocity=False, velocity_gap=5, velocity_start=1):
    """
    Create inputs and labels for training.
    """

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

    assert velocity_start < time_steps, "first velocity command occurs after last time-step"

    ix = np.random.randint(low=0, high=state_size, size=n_samples)
    input = make_bumps(ix)
    batch_size = len(ix)

    if noise:
        # sample noisy positions, sample noise for those positions, add noise to inputs
        assert 0 <= noise_density <= 1, "Density is not between 0 and 1"
        assert 0 <= noise_intensity <= 1, "Intensity is not between 0 and 1"
        max_noise = np.amax(input[0]) * noise_intensity
        noise = np.random.uniform(low=0, high=max_noise, size=input.shape)

        inactive_mask = input == 0
        noise_sample = np.random.uniform(size=noise.shape)
        noise[noise_sample < noise_density] *= 0  # take density % of noise
        noise *= inactive_mask  # remove noise from true activity
        input_noise = input + noise
        input_noise = sum_normalize(input_noise)
    else:
        input_noise = input

    # pad the inputs for time. inputs are batch X time X STATE_SIZE
    inputs = np.expand_dims(input_noise, axis=1)
    pad = np.zeros((batch_size, time_steps - 1, inputs.shape[2]))
    inputs = np.concatenate((inputs, pad), axis = 1)

    if velocity:
        vel_ix = np.arange(velocity_start, time_steps, velocity_gap)
        vel_options = np.array([-1, 1])
        vel_per_batch = len(vel_ix)
        one_hot_len = len(vel_options)
        vel_total = vel_per_batch * batch_size

        vel_direction = np.random.randint(len(vel_options), size=vel_total)
        vel_one_hot = np.zeros((vel_total, one_hot_len))
        vel_one_hot[np.arange(vel_total), vel_direction] = 1
        # vel_one_hot = np.expand_dims(vel_one_hot, axis=1)
        vel_one_hot = vel_one_hot.reshape(batch_size, vel_per_batch, one_hot_len)

        shift = vel_options[vel_direction]
        shift = shift.reshape(batch_size, vel_per_batch)
        # concatenate states with velocities
        velocity = np.zeros((inputs.shape[0], inputs.shape[1], one_hot_len))
        velocity[:, vel_ix, :] = vel_one_hot

        inputs = np.concatenate([inputs, velocity], axis=2)
        # create labels from input
        labels = [create_labels(x, s, vel_ix) for x, s in zip(input, shift)]
        labels = np.stack(labels, axis=0)
    else:
        labels = [create_labels(x, [], []) for x in input]
        labels = np.stack(labels, axis=0)
        vel_per_batch = 0

    return inputs, labels, vel_per_batch


if __name__ == '__main__':
    STATE_SIZE = 10
    batch = 3
    inputs, labels, _ = create_inputs(10, state_size = 10, time_steps=50, bump_size=2,
                                          noise= False,
                                          velocity=True, velocity_start=4, velocity_gap = 5)

    print(inputs.shape)
    print(labels.shape)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    plt.axis('off')
    ax[0].set_title('Input')
    sns.heatmap(inputs[0], vmin =0, vmax = 1, cbar=False, ax = ax[0])
    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax[0].yaxis.set_major_formatter(ticker.ScalarFormatter())

    ax[1].set_title('Label')
    sns.heatmap(labels[0], vmin=0, vmax=1, cbar=True, ax = ax[1])
    ax[1].yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax[1].yaxis.set_major_formatter(ticker.ScalarFormatter())

    plt.show()
