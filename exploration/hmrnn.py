import tensorflow as tf
import numpy as np


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

    def run():
        init = tf.initialize_all_variables()
        session = tf.Session()
        session.run(init)

    def get_operation(self, z, z_below):
        # TODO: Make sure this is actually checking for equality
        if z == 1:
            return 'flush'

        if z == 0 and z_below == 1:
            return 'update'

        if z == 0 and z_below == 0:
            return 'copy'

        raise ValueError('Invalid indicators')

    def hmlstm_layer(self):
        # figure out operation
        operation = self.get_operation(self.z, self.z_below)

        if operation == 'copy':
            return self.c, self.h, self.z

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
        z_tilde = tf.contrib.keras.backend.hard_sigmoid(joint_input[-1])

        new_z = tf.constant(np.random.binomial(1, z_tilde))

        if operation == 'update':
            new_c = tf.add(tf.multiply(f, self.c), tf.multiply(i, g))

        if operation == 'flush':
            new_c = tf.multiply(i, g)

        new_h = tf.multiply(o, tf.tanh(new_c))
        return new_c, new_h, new_z

    def hmlstm_module(self):
        for l in self.layers:
            placeholders = {
                self.c: 1,
                self.h: 1,
                self.below: 1,
                self.h_above: 1,
                self.z: 1,
                self.z_below: 1,
            }
            session.run(
                [], )
        lowest_c, lowest_h, lowest_z = self.hmlstm_layer(tf.constant())

    def output_module(inputs):
        pass


x = np.linspace(0, 300, 10000)
signal = np.sin(x * 3) + np.sin(x * .5) * 5 + np.sin(x * .1) * -2

init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)

for t in range(len(signal)):
    lowest_c, lowest_h, lowest_z = hmlstm_layer()
