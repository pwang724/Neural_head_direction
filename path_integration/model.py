import numpy as np
import tensorflow as tf
from config import Options
import utils


class RNN:

    def __init__(self, opts):
        # make weights, placeholders and training ops
        timesteps = opts.agent_steps
        if opts.test and opts.test_type == '2-leg':
            timesteps = opts.r0 + opts.r1
        if opts.use_gap:
            timesteps = timesteps * (opts.step_gap+1)

        state_size = opts.network_state_size
        if opts.angle_format == 'trig':
            d_angle = 1
        else:
            d_angle = opts.d_angle

        d_out = 2
        d_step = opts.d_step
        weights_name = opts.weights_name
        network_loss = opts.network_loss

        dt = .2
        tau = .2
        self.time_const = dt / tau
        self.continuous = opts.continuous
        self.positive_input = opts.positive_input
        self.inputs = tf.placeholder(tf.float32, [None, timesteps, d_angle + d_step], name="input_placeholder")
        self.labels = tf.placeholder(tf.float32, [None, timesteps, d_out], name="labels_placeholder")
        self.lr = tf.placeholder(tf.float32, [], name='lr')
        if weights_name:  # load pretrained weights
            # load state size from input weights as well
            W_ah, W_sh, W_hh, W_out, bias = utils.load_weights(opts.folder, opts.weights_name)
            self.W_ah = tf.Variable(W_ah, "angle_weights")
            self.W_sh = tf.Variable(W_sh, "step_weights")
            self.W_hh = tf.Variable(W_hh, "hidden_weights")
            self.W_out = tf.Variable(W_out, "output_weights")
            self.bias = tf.Variable(bias, "bias")
        else:
            self.W_ah = tf.get_variable("angle_weights", [d_angle, state_size])
            self.W_sh = tf.get_variable("step_weights", [d_step, state_size])
            self.W_hh = tf.get_variable("hidden_weights", [state_size, state_size])
            self.W_out = tf.get_variable("output_weights", [state_size, 2])  # output an angle in rad or deg, distance
            self.bias = tf.get_variable("bias", [1, state_size])

        self.W_xh = tf.concat([self.W_ah, self.W_sh], axis=0)  # Wxh is trained - split by input dimensions to analyze
        self.init_state = tf.placeholder(tf.float32, [None, state_size], name='init_state')
        self.W_hh_mask = tf.constant(1 - np.eye(state_size).astype(np.float32))  # identity mask
        self.W_hh_masked = self.W_hh * self.W_hh_mask

        type = opts.rnn_type
        types = ['sigmoid', 'tanh', 'relu', 'relu6']
        assert type in types, "Invalid nonlinearity"
        if type == 'sigmoid':
            self.fn = tf.nn.sigmoid
        elif type == 'tanh':
            self.fn = tf.tanh
        elif type == 'relu':
            self.fn = tf.nn.relu
        else:  # type == 'relu6'
            self.fn = tf.nn.relu6

        labels_series = tf.unstack(self.labels, axis=1)  # unstack across time
        self.states = tf.scan(self.scan_fn, tf.transpose(self.inputs, [1,0,2]), initializer=self.init_state)
        state_series = tf.unstack(self.states, axis=0)  # output of scan leaves t on zero axis - separate by time
        self.predictions = [tf.matmul(logits, self.W_out) for logits in state_series]

        if network_loss:
            alpha_activity_loss = .001  # penalty for activity
            alpha_weight_loss = .001  # penalty for making unnecessary connections
        else:
            alpha_activity_loss = 0
            alpha_weight_loss = 0


        activity_loss = alpha_activity_loss * tf.reduce_mean(self.states)
        # activity_loss = alpha_activity_loss * tf.reduce_mean(tf.square(self.states))
        weights_loss = alpha_weight_loss * (tf.reduce_mean(tf.square(self.W_hh)) + tf.reduce_mean(tf.square(self.W_xh)))

        # separate way
        if opts.separate_losses:
            angle_amp = 100
            self.ang_loss = [tf.losses.mean_squared_error(labels=y[:, 0], predictions=yhat[:, 0])
                             for y, yhat in zip(labels_series, self.predictions)]
            self.dist_loss = [tf.losses.mean_squared_error(labels=y[:, 1], predictions=yhat[:, 1])
                              for y, yhat in zip(labels_series, self.predictions)]
            self.total_loss = angle_amp*tf.reduce_mean(self.ang_loss) + tf.reduce_mean(self.dist_loss) + activity_loss + weights_loss
        else:
            pred_loss = [tf.losses.mean_squared_error(labels=y, predictions=yhat)
                         for y, yhat in zip(labels_series, self.predictions)]
            self.total_loss = tf.reduce_mean(pred_loss) + activity_loss + weights_loss

        # self.train_op = tf.train.AdamOptimizer(.001).minimize(self.total_loss)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss)

        if self.positive_input:
            self.W_ah_trained = tf.abs(self.W_xh[:d_angle, :])
        else:
            self.W_ah_trained = self.W_xh[:d_angle, :]
        self.W_sh_trained = self.W_xh[d_angle:, :]

    def scan_fn(self, hprev, input):
        state_update = tf.matmul(hprev, self.W_hh * self.W_hh_mask)
        input_update = tf.matmul(input, self.W_xh)
        # discrete time rnn
        h = self.fn(state_update + input_update + self.bias)
        # continuous time rnn
        if self.continuous:
            h = (1 - self.time_const) * hprev + self.time_const * h
        return h

    def batch_inputs(self, inputs, labels, opts):
        """Wrap the inputs in tensorflow batches."""
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        if opts.test:
            dataset = dataset.batch(tf.cast(opts.batch_size, tf.int64))
        else:
            dataset = dataset.shuffle(int(1E6)).batch(tf.cast(opts.batch_size, tf.int64))  # inferred repeat

        train_iter = dataset.make_initializable_iterator()
        next_element = train_iter.get_next()
        return train_iter, next_element


if __name__ == '__main__':
    opts = Options()
    RNN(opts)