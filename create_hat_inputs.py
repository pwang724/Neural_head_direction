import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# set seed for reproducibility
# np.random.seed(2)


class velocityTraining:

    def __init__(self, state_size):
        self.state_size = state_size


    def sum_normalize(self, x):
        """Normalize by sum row-wise."""
        if len(x.shape) > 1:
            x = (x.T / np.sum(x, axis=1)).T

        else:
            # Vector
            x = x / np.sum(x)

        return x

    def create_inputs(self, ix, timesteps=5, spread=1, dev=1, create_noise=False, intensity=.5, density=.5,
                      velocity_gap=5, velocity_start=1):
        """
        Create inputs and labels for training.
        """
        assert velocity_start < timesteps, "Velocity commands must be delivered before last timestep."
        spread = min(spread, self.state_size)
        if (spread % 2) == 0:
            spread += 1

        # create starting inputs
        span = int(np.floor(spread / 2))
        activity = self.sum_normalize(norm.pdf(np.arange(-span, span + 1), scale=dev))

        denoised_inputs = np.zeros((len(ix), self.state_size))
        denoised_inputs[:, :spread] += activity
        denoised_inputs = np.stack([np.roll(ipt, s) for ipt, s in zip(denoised_inputs, ix-span)], axis=0)

        if create_noise:
            # sample noisy positions, sample noise for those positions, add noise to inputs
            assert 0 <= density <= 1, "Density is not between 0 and 1"
            assert 0 <= intensity <= 1, "Intensity is not between 0 and 1"
            max_noise = np.amax(denoised_inputs[0]) * intensity
            noise = np.random.uniform(low=0, high=max_noise, size=denoised_inputs.shape)

            inactive_mask = denoised_inputs == 0
            noise_sample = np.random.uniform(size=noise.shape)
            noise[noise_sample < density] *= 0  # take density % of noise
            noise *= inactive_mask  # remove noise from true activity

            unpadded_inputs = denoised_inputs + noise
            unpadded_inputs = self.sum_normalize(unpadded_inputs)
        else:
            unpadded_inputs = denoised_inputs

        # pad the inputs for time
        inputs = unpadded_inputs[:, np.newaxis, :]
        input_pad = [inputs] + [np.zeros_like(inputs)] * (timesteps - 1)
        inputs = np.concatenate(input_pad, axis=1)

        # make velocity commands
        vel_ix = np.arange(velocity_start, timesteps, velocity_gap)
        vel_options = np.array([-1, 1])
        vel_direction = np.random.randint(len(vel_options), size=vel_ix.shape[0] * inputs.shape[0])
        shift = np.array([vel_options[d] for d in vel_direction])
        shift = shift.reshape(inputs.shape[0], vel_ix.shape[0])  # reshape to n_inputs, n_shifts

        _velocity = np.stack([np.roll(np.array([1, 0]), d) for d in vel_direction])
        _velocity = np.reshape(_velocity, (inputs.shape[0], vel_ix.shape[0], 2))

        # concatenate states with velocities
        velocity = np.zeros((inputs.shape[0], inputs.shape[1], 2))
        velocity[:, vel_ix, :] = _velocity
        inputs = np.concatenate([inputs, velocity], axis=2)

        # create labels from denoised inputs
        labels = [self.create_labels(x, s, vel_ix, timesteps) for x, s in zip(denoised_inputs, shift)]
        labels = np.stack(labels, axis=0)
        return inputs, labels, vel_ix.shape[0]

    def create_labels(self, input, shift, ix, timesteps):
        """
        Create labels for a single starting state with no noise.
        :param input: 1D input slice
        :param shift: velocity commands to be applied to state at previous timestep.
        :param ix: indices where each velocity command is applied
        :param timesteps: total trial length
        :return: labels
        """
        # take the first state, rotate it using shift
        labels = np.zeros((timesteps, self.state_size))
        labels += input
        shifted = [input]
        for i, s in enumerate(shift):
            shifted.append(np.roll(shifted[-1], s))
            labels[ix[i]:] *= 0
            labels[ix[i]:] += shifted[-1]

        return labels


if __name__ == '__main__':
    STATE_SIZE = 10
    batch = 2
    ix = np.random.randint(low=0, high=STATE_SIZE, size=1)
    vel = velocityTraining(STATE_SIZE)
    inputs, labels = vel.create_inputs(ix, timesteps=100, spread=3, velocity_start=4)

    print(inputs)
    print(labels)
    plt.figure()
    sns.heatmap(inputs[0])
    plt.figure()
    sns.heatmap(labels[0])
    # dist = pd.DataFrame(np.random.randn(7, 3), columns=['A', 'B', 'C'])
    # print(dist)
    # sns.barplot(dist)

    # sns.set(style="whitegrid")
    # tips = sns.load_dataset("tips")
    # print(tips)
    # ax = sns.barplot(x="day", y="total_bill", data=tips)
    plt.show()
