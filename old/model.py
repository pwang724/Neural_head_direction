import tensorflow as tf
import os
import pickle as pkl
from old import rnn as rnn_helper
import config


class Model(object):
    '''abstract model class'''
    def __init__(self, save_path):
        '''make model'''
        if save_path is None:
            save_path = os.getcwd()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = save_path
        self.saver = None
        self.opts = None
        self.weight_dict = None

    def save(self, path = None):
        if path is not None:
            save_path = path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        else:
            save_path = self.save_path
        save_path = os.path.join(save_path, 'model.ckpt')
        sess = tf.get_default_session()
        save_path = self.saver.save(sess, save_path)
        print("[***] Model saved in path: %s" % save_path)

    def load(self):
        save_path = self.save_path
        save_path = os.path.join(save_path, 'model.ckpt')
        sess = tf.get_default_session()
        self.saver.restore(sess, save_path)
        print("[***] Model restored from path: %s" % save_path)

    def save_weights(self, path = None):
        '''save model using pickle'''
        if path is not None:
            save_path = path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        else:
            save_path = self.save_path

        f_name = os.path.join(save_path, self.opts.weight_name)
        sess = tf.get_default_session()
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # vars = tf.all_variables()
        # vars = self.weight_dict
        var_dict = {v.name: sess.run(v) for v in vars}
        with open(f_name + ".pkl", 'wb') as f:
            pkl.dump(var_dict, f)
        print("[***] Model weights saved in path: %s" % save_path)


class RNN(Model):
    """An RNN made to model 1D attractor network"""
    def __init__(self, X_pl, Y_pl, opts, training=True):
        super(RNN, self).__init__(opts.save_path)

        self.opts = opts
        with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
            self._build(X_pl, Y_pl)

        if training:
            learning_rate = opts.learning_rate
            optimizer= tf.train.AdamOptimizer(learning_rate)
            excludes = []
            k = config.weight_names()

            if not opts.stationary and opts.fix_weights:
                name = 'model/hidden/' + k.W_h_aa + ':0'
                W_h_aa = [v for v in tf.global_variables() if v.name == name]
                excludes += W_h_aa

            trainable_list = [v for v in tf.trainable_variables() if v not \
                in excludes]

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.total_loss,
                                                   var_list= trainable_list)
            self.saver = tf.train.Saver()
            print('[***] Training Variables:')
            for v in trainable_list:
                print(v)

    def _build(self, x, y):
        opts = self.opts
        stationary = opts.stationary
        state_size = opts.state_size
        rnn_size = opts.rnn_size
        batch_size = opts.batch_size

        time_loss_start = opts.time_loss_start
        time_loss_end= opts.time_loss_end

        inputs_series = tf.unstack(x, axis=1)
        labels_series = tf.unstack(y, axis=1)
        W_h = rnn_helper.define_weights(opts)

        init_state = tf.zeros(shape=[batch_size, rnn_size], dtype= tf.float32)
        state_series = [init_state]
        logit_series = []
        for i, current_input in enumerate(inputs_series):
            next_state, next_logit = rnn_helper.rnn(state_series[-1], current_input, i, opts)
            state_series.append(next_state)
            logit_series.append(next_logit)

        state_series.pop(0)
        self.states = state_series
        self.logits = logit_series
        self.predictions = [tf.nn.softmax(l) for l in self.logits]
        xe_losses = [tf.nn.softmax_cross_entropy_with_logits_v2(logits=l, labels=labels)
                     for l, labels in zip(self.logits, labels_series)]
        self.xe_loss = tf.reduce_mean(xe_losses[time_loss_start:time_loss_end])

        W_h_aa = W_h[:state_size, :state_size]
        W_h_ab = W_h[:, state_size:]
        W_h_ba = W_h[state_size:,:]
        W_h_bb = W_h[state_size:, state_size:]
        self.weight_loss = opts.weight_alpha * \
                           (tf.reduce_mean(tf.abs(W_h_ab))
                            + tf.reduce_mean(tf.abs(W_h_ba))
                            + tf.reduce_mean(tf.abs(W_h_bb))
                            )

        rnn_activity = tf.stack(state_series, axis=1)
        extra_neurons_activity = rnn_activity[:,:,state_size:]
        self.activity_loss = opts.activity_alpha * \
                             tf.reduce_mean(extra_neurons_activity)
        self.total_loss = self.xe_loss + self.weight_loss + self.activity_loss
