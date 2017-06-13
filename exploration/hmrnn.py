import tensorflow as tf
import numpy as np

# TODO: Make sure conditionals all work (types match)
# TODO: allow for different size input and hidden state


class hmlstm(object):
    def __init__(self, layers=3, state_size=5):
        self.state_size = state_size
        self.c = tf.placeholder(tf.float32, shape=(state_size, 1))
        self.h = tf.placeholder(tf.float32, shape=(state_size, 1))
        self.h_below = tf.placeholder(tf.float32, shape=(state_size, 1))
        self.h_above = tf.placeholder(tf.float32, shape=(state_size, 1))
        self.z = tf.placeholder(tf.float32, shape=(1))
        self.z_below = tf.placeholder(tf.float32, shape=(1))
        self.num_layers = layers
        self.layers = [self.hmlstm_layer() for _ in range(layers)]
        self.hidden_states = tf.placeholder(tf.float32, shape=(layers, state_size))
        self.prediction = self.output_module()

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

        # TODO: Make sure this condition actually works
        new_z = tf.cond(
            tf.greater(
                tf.squeeze(z_tilde),
                tf.squeeze(tf.constant(.5, dtype=tf.float32))),
            lambda: tf.ones(1), lambda: tf.zeros(1))

        # copy is handled in the run method
        def update():
            return tf.add(tf.multiply(f, self.c), tf.multiply(i, g))

        def flush():
            return tf.multiply(i, g)

        # TODO: Make sure this condition actually works
        new_c = tf.cond(
            tf.equal(
                tf.squeeze(self.z),
                tf.squeeze(tf.constant(1, dtype=tf.float32))), flush, update)

        new_h = tf.multiply(o, tf.tanh(new_c))
        return new_c, new_h, new_z

    def run(self, signal, epochs=100):
        session = tf.Session()
        init = tf.global_variables_initializer()
        session.run(init)

        for _ in range(epochs):

            # for first run
            ones = np.ones((self.state_size, 1))
            last_run = [(ones, ones, np.ones(1)) for _ in self.layers]
            current_run = [(1., 1., 1.)] * len(self.layers)
            for t, s in enumerate(signal):

                for i, l in enumerate(self.layers):
                    # short circut copy operation
                    if last_run[i][2] == 0. and current_run[i - 1][2] == 0.:
                        current_run[i] = last_run[i]
                        continue

                    placeholders = self.get_placeholders(
                        last_run, current_run, i, s)

                    current_run[i] = session.run(l, placeholders)

                hidden_states = np.array([h[1][:, 0] for h in current_run])
                prediction = session.run([self.prediction],
                                         {self.hidden_states: hidden_states})

                # TODO: Calculate prediction loss

                last_run = current_run

    def get_placeholders(self, last_run, current_run, i, s):
        if i == len(self.layers) - 1:
            # for top layer, the hidden state from above is zero
            ha = np.zeros((self.state_size, 1))
        else:
            ha = last_run[i + 1][1]

        placeholders = {
            self.c: last_run[i][0],
            self.h: last_run[i][1],
            self.z: last_run[i][2],
            self.h_below: s.reshape(-1, 1) if i == 0. else last_run[i - 1][1],
            self.h_above: ha,
            self.z_below: np.ones(1) if i == 0. else last_run[i - 1][2],
        }
        return placeholders

    def output_module(self):
        # inputs are concatenated output from all hidden layers
        # assume they come in L x state_size

        # gates
        init_weights = np.random.rand(self.num_layers,
                                      self.state_size * self.num_layers)
        gate_weights = tf.Variable(init_weights, dtype=tf.float32)
        col_inputs = tf.reshape(self.hidden_states, (self.num_layers * self.state_size, 1))
        gates = tf.sigmoid(tf.matmul(gate_weights, col_inputs))
        gated = tf.multiply(gates, self.hidden_states)

        # embedding
        embedding_size = 100
        em_init_weights = np.random.rand(self.state_size, embedding_size)
        embedding_weights = tf.Variable(em_init_weights, dtype=tf.float32)
        embedding = tf.nn.relu(
            tf.reduce_sum(tf.matmul(gated, embedding_weights), axis=0))
        col_embedding = tf.reshape(embedding, (embedding_size, 1))

        # feed forward network
        hidden_size = 200
        output_size = 29  # alphanumeric, period, comma, space

        # first layer
        b1 = tf.Variable(np.random.rand(hidden_size, 1), dtype=tf.float32)
        w1 = tf.Variable(
            np.random.rand(hidden_size, embedding_size), dtype=tf.float32)
        l1 = tf.nn.tanh(tf.matmul(w1, col_embedding) + b1)

        # second layer
        b2 = tf.Variable(np.random.rand(hidden_size, 1), dtype=tf.float32)
        w2 = tf.Variable(
            np.random.rand(hidden_size, hidden_size), dtype=tf.float32)
        l2 = tf.nn.tanh(tf.matmul(w2, l1) + b2)

        # output
        b3 = tf.Variable(np.random.rand(output_size, 1), dtype=tf.float32)
        w3 = tf.Variable(
            np.random.rand(output_size, hidden_size), dtype=tf.float32)
        prediction = tf.nn.softmax(tf.matmul(w3, l2) + b3)
        return prediction
