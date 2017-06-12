import tensorflow as tf
import numpy as np
import scipy.stats as st

# TODO: allow for different size input and hidden state


class hmlstm(object):
    def __init__(self, layers=3, state_size=200):
        self.state_size = state_size
        self.c = tf.placeholder(tf.float32, shape=(state_size, 1))
        self.h = tf.placeholder(tf.float32, shape=(state_size, 1))
        self.h_below = tf.placeholder(tf.float32, shape=(state_size, 1))
        self.h_above = tf.placeholder(tf.float32, shape=(state_size, 1))
        self.z = tf.placeholder(tf.float32, shape=(1))
        self.z_below = tf.placeholder(tf.float32, shape=(1))
        self.layers = [self.hmlstm_layer() for _ in range(layers)]

    def hmlstm_layer(self):
        # set biases and weights
        weight_size = (self.state_size * 4) + 1
        b = tf.Variable(np.zeros((weight_size, 1)), dtype=tf.float32)

        def initialize_weights():
            return np.random.rand(weight_size, self.state_size)

        w = tf.Variable(initialize_weights(), dtype=tf.float32)
        wa = tf.Variable(initialize_weights(), dtype=tf.float32)
        wb = tf.Variable(initialize_weights(), dtype=tf.float32)

        # process gates
        joint_input = tf.add(
            tf.add(
                tf.matmul(w, self.h),
                tf.multiply(self.z, tf.matmul(wa, self.h_above))),
            tf.add(tf.multiply(self.z_below, tf.matmul(wb, self.h_below)), b))

        f = tf.sigmoid(joint_input[:self.state_size])
        i = tf.sigmoid(joint_input[self.state_size:2 * self.state_size])
        o = tf.sigmoid(joint_input[2 * self.state_size:3 * self.state_size])
        g = tf.tanh(joint_input[3 * self.state_size:4 * self.state_size])
        z_tilde = tf.sigmoid(joint_input[-1:])

        # TODO: FIX CONDITIONAL
        print(z_tilde.get_shape())
        new_z = tf.case(
            {
                tf.greater(z_tilde[0], tf.constant(.5, dtype=tf.float32)[0]):
                lambda: np.ones(1),
                tf.less_equal(z_tilde[0], tf.constant(.5, dtype=tf.float32)[0]):
                lambda: np.zeros(1),
            }, default=lambda: np.zeros(1)
        )

        # copy is handled in the run method
        def update():
            return tf.add(tf.multiply(f, self.c), tf.multiply(i, g))

        def flush():
            return tf.multiply(i, g)

        # TODO: Make sure this condition actually works
        new_c = tf.cond(tf.equal(self.z, tf.placeholder(1)), flush, update)

        new_h = tf.multiply(o, tf.tanh(new_c))
        return new_c, new_h, new_z

    def run(self, signal, epochs=100):
        init = tf.global_variables_initializer()
        session = tf.Session()
        session.run(init)

        # for first run
        ones = np.ones((self.state_size, 1))
        last_run = [(ones, ones, np.ones(1)) for _ in self.layers]
        for _ in range(epochs):

            for t, s in enumerate(signal):
                current_run = []
                print(t)

                for i, l in enumerate(self.layers):
                    # short circut copy operator
                    if last_run[i][2] == 0 and current_run[i - 1][2] == 0:
                        current_run[i] = last_run[i]
                        continue

                    if i == len(self.layers) - 1:
                        ha = np.zeros((self.state_size, 1))
                    else:
                        ha = last_run[i + 1][1]

                    placeholders = {
                        self.c:
                        last_run[i][0],
                        self.h:
                        last_run[i][1],
                        self.z:
                        last_run[i][2],
                        self.h_below:
                        s.reshape(-1, 1) if i == 0 else last_run[i - 1][1],
                        self.h_above:
                        ha,
                        self.z_below:
                        np.ones(1) if i == 0 else last_run[i - 1][2],
                    }

                    current_run[i] = session.run(l, placeholders)

                last_run = current_run

    def output_module(inputs):
        pass
