import tensorflow as tf





# def gru(self, h_prev, input, name):
#     #  update gate
#     z = tf.sigmoid(input + tf.matmul(h_prev, self.W_h_tf[0]))
#     #  reset gate
#     r = tf.sigmoid(input + tf.matmul(h_prev, self.W_h_tf[1]))
#     #  intermediate
#     h = tf.tanh(input + tf.matmul((r * h_prev), self.W_h_tf[2]), name='time_{}'.format(name))
#     # new state
#     st = (1 - z) * h + (z * h_prev)
#     return st