import tensorflow as tf
from old.abstract_model import Model
import numpy as np
EPS = np.finfo(float).eps


class RNN(Model):
    def __init__(self, data, opts, training=True, lesion_ix=None, perturb=None):
        super(RNN, self).__init__(opts.save_path)

        X_pl, Y_pl, N_pl = data[0], data[1], data[2]
        self.opts = opts
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            self._build(X_pl, Y_pl, N_pl, lesion_ix, perturb)

        print('built')

        if training:
            optimizer = tf.train.AdamOptimizer(opts.learning_rate)
            excludes = []
            trainable_list = [v for v in tf.trainable_variables() if v not in excludes]

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.total_loss, var_list=trainable_list)
                self.grad = optimizer.compute_gradients(self.error_loss, [self._Whh])

            print('[***] Training Variables:')
            for v in trainable_list:
                print(v)

        self.saver = tf.train.Saver()

    def _build(self, x, y, n, lesion_ix, perturb):
        print('build start')
        opts = self.opts
        rnn_size = opts.rnn_size
        time_loss_start = opts.time_loss_start
        time_loss_end = opts.time_loss_end
        batch_size = opts.batch_size
        output_mode = opts.output_mode

        assert opts.activation_fn in ['relu', 'tanh', 'relu6', 'retanh', 'sigmoid'], "Invalid nonlinearity"
        fn = opts.activation_fn
        if fn == 'sigmoid':
            self.activation_fn = tf.nn.sigmoid
        elif fn == 'tanh':
            self.activation_fn = tf.tanh
        elif fn == 'relu':
            self.activation_fn = tf.nn.relu
        elif fn == 'relu6':
            self.activation_fn = tf.nn.relu6
        else:
            self.activation_fn = lambda L: tf.nn.relu(tf.nn.tanh(L))  # a rectified tanh function

        if lesion_ix is not None:
            lesion = np.zeros((batch_size, rnn_size))
            lesion[:, lesion_ix] = -1000
            self.lesion_activity = tf.convert_to_tensor(lesion, dtype=tf.float32)
            self.lesion = True
        else:
            self.lesion = False

        inputs_series = tf.unstack(x, axis=1)
        labels_series = tf.unstack(y, axis=1)
        noise_series = tf.unstack(n, axis=1)

        if perturb is not None:
            self.perturb = True
            self.perturb_series = tf.unstack(tf.convert_to_tensor(perturb, dtype=tf.float32), axis=1)
        else:
            self.perturb = False

        init_state = tf.zeros(shape=[batch_size, rnn_size], dtype=tf.float32)
        state_series = [init_state]
        logit_series = []

        self._initialize_vars(x, y)
        for i, (current_input, noise) in enumerate(zip(inputs_series, noise_series)):
            next_state, next_logit = self._scan_fn(state_series[-1], current_input, noise, i)
            state_series.append(next_state)
            logit_series.append(next_logit)

        state_series.pop(0)
        rnn_activity = tf.stack(state_series, axis=1)
        self.activity_loss = opts.activity_alpha * tf.reduce_mean(tf.square(rnn_activity))
        self.weight_loss = opts.weight_alpha * tf.reduce_mean(tf.square(self._Whh))

        self.states = tf.stack(state_series, axis=1)
        self.logits = tf.stack(logit_series, axis=1)
        if output_mode == 'bump':
            error = [tf.nn.softmax_cross_entropy_with_logits_v2(labels=lab, logits=log)
                   for lab, log in zip(labels_series, logit_series)]
            self.predictions = tf.nn.softmax(self.logits, axis=2)
        else:
            error = [tf.losses.mean_squared_error(predictions=log, labels=labels)
                   for log, labels in zip(logit_series, labels_series)]
            self.predictions = self.logits

        self.error_loss = tf.reduce_mean(error[time_loss_start:time_loss_end])
        self.total_loss = self.error_loss + self.weight_loss + self.activity_loss

    def _initialize_vars(self, x, y):
        opts = self.opts
        EI_in = opts.EI_in
        EI_h = opts.EI_h
        EI_out = opts.EI_out
        prop_ex = opts.prop_ex  # proportion excitatory
        assert 0 <= prop_ex <= 1, "Proportion excitatory neurons must be between 0 and 1."
        rnn_size = opts.rnn_size
        nE = int(prop_ex * rnn_size)
        nI = rnn_size - nE
        self._Wxh = tf.get_variable("input_weights", [x.shape[-1], rnn_size])

        if EI_h or EI_out:  # EI mask
            ei_mask = np.eye(rnn_size)
            ei_mask[nE:] *= -1
            self.ei_mask = tf.convert_to_tensor(ei_mask, dtype=tf.float32)

        if EI_h:
            # balance EI recurrent connections to avoid activity blowup
            # GAMMA DISTRIBUTION: scale=alpha (k), scale=beta (theta), mean=alpha/beta (k*theta)
            target = 1
            alpha = 2
            E = tf.random.gamma([nE, rnn_size], alpha, beta=nE * alpha / (target+EPS))
            I = tf.random.gamma([nI, rnn_size], alpha, beta=nI * alpha / (target+EPS))
            self._Whh = tf.Variable(tf.concat([E, I], axis=0), name="hidden_weights")
            print(f'{nE} excitatory, {rnn_size - nE} inhibitory')
        else:
            self._Whh = tf.get_variable("hidden_weights", [rnn_size, rnn_size])

        self._Wout = tf.get_variable("output_weights", [rnn_size, y.shape[-1]])
        self.Whh_mask = 1 - tf.eye(rnn_size)
        self.Wh_bias = tf.get_variable("hidden_bias", [1, rnn_size], initializer=tf.zeros_initializer)
        self.Wout_bias = tf.get_variable("output_bias", [1, y.shape[-1]], initializer=tf.zeros_initializer)

    def get_connections(self):
        opts = self.opts
        if opts.EI_in:
            Wxh = tf.abs(self._Wxh)
        else:
            Wxh = self._Wxh

        if opts.EI_h:
            Whh = tf.matmul(self.ei_mask, tf.abs(self._Whh))
        else:
            Whh = self._Whh

        if opts.mask:
            Whh *= self.Whh_mask

        if opts.EI_out:
            Wout = tf.matmul(self.ei_mask, tf.abs(self._Wout))
        else:
            Wout = self._Wout

        return Wxh, Whh, Wout

    def _scan_fn(self, prev_state, input, noise, t):
        opts = self.opts
        Wxh, Whh, Wout = self.get_connections()
        hidden_act = tf.matmul(prev_state, Whh) + tf.matmul(input, Wxh) + self.Wh_bias
        if opts.noise:
            hidden_act += noise
            # noise = opts.noise_intensity * tf.random_normal([opts.batch_size, opts.rnn_size])
            # noise_mask = tf.cast(tf.random.uniform([opts.batch_size, opts.rnn_size]) < opts.noise_intensity, tf.float32)
            # hidden_act += opts.noise_intensity * noise * noise_mask

        if self.lesion:
            hidden_act += self.lesion_activity

        if self.perturb:
            hidden_act += self.perturb_series[t]

        state = self.activation_fn(hidden_act)
        logit = tf.matmul(state, Wout) + self.Wout_bias
        return [state, logit]


