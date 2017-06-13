from tensorflow.python.framework import ops
from string import ascii_lowercase
import tensorflow as tf
import numpy as np
import re

# TODO: Enable input different from hidden state size
# TODO: Make sure conditionals all work (types match)


class hmlstm(object):
    def __init__(self,
                 layers=3,
                 state_size=29,
                 batch_size=50,
                 step_size=1,
                 output_size=29):
        self.state_size = state_size
        self.batch_size = batch_size
        self.step_size = step_size
        self.output_size = output_size
        self.c = tf.placeholder(tf.float32, shape=(state_size, 1))
        self.h = tf.placeholder(tf.float32, shape=(state_size, 1))
        self.h_below = tf.placeholder(tf.float32, shape=(state_size, 1))
        self.h_above = tf.placeholder(tf.float32, shape=(state_size, 1))
        self.z = tf.placeholder(tf.float32, shape=())
        self.z_below = tf.placeholder(tf.float32, shape=())
        self.iteration = tf.placeholder(tf.float32, shape=())
        self.num_layers = layers
        self.layers = [self.hmlstm_layer() for _ in range(layers)]
        self.hidden_states = tf.placeholder(
            tf.float32, shape=(layers, state_size))
        self.prediction = self.output_module()
        self.current_output = tf.placeholder(
            tf.float32, shape=(output_size, 1))
        self.current_loss = tf.placeholder(tf.float32, shape=())
        self.loss = self.calculate_loss()
        self.train = self.minimize_loss()

    def hmlstm_layer(self):
        # create bias and weight variables
        weight_size = (self.state_size * 4) + 1
        b = tf.Variable(np.zeros((weight_size, 1)), dtype=tf.float32)

        def initialize_weights():
            return np.random.rand(weight_size, self.state_size)

        w = tf.Variable(initialize_weights(), dtype=tf.float32)
        wa = tf.Variable(initialize_weights(), dtype=tf.float32)
        wb = tf.Variable(initialize_weights(), dtype=tf.float32)

        # calculate LSTM-like gates
        joint_input = tf.add(
            tf.add(
                tf.matmul(w, self.h),
                tf.multiply(self.z, tf.matmul(wa, self.h_above))),
            tf.add(tf.multiply(self.z_below, tf.matmul(wb, self.h_below)), b))

        f = tf.sigmoid(joint_input[:self.state_size])
        i = tf.sigmoid(joint_input[self.state_size:2 * self.state_size])
        o = tf.sigmoid(joint_input[2 * self.state_size:3 * self.state_size])
        g = tf.tanh(joint_input[3 * self.state_size:4 * self.state_size])

        # use slope annealing trick
        slope_multiplier = .02 + (self.iteration / 1e5)
        z_tilde = tf.sigmoid(joint_input[-1:] * slope_multiplier)

        # replace gradient calculation - use straight-through estimator
        # see: https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
        graph = tf.get_default_graph()
        with ops.name_scope('BinaryRound') as name:
            with graph.gradient_override_map({'Round': 'Identity'}):
                new_z = tf.round(z_tilde, name=name)

        # copy is handled in the run method
        def update():
            return tf.add(tf.multiply(f, self.c), tf.multiply(i, g))

        def flush():
            return tf.multiply(i, g)

        # calculate new cell and hidden states
        new_c = tf.cond(
            tf.equal(
                tf.squeeze(self.z),
                tf.squeeze(tf.constant(1, dtype=tf.float32))), flush, update)
        new_h = tf.multiply(o, tf.tanh(new_c))

        return new_c, new_h, tf.squeeze(new_z)

    def output_module(self):
        # inputs are concatenated output from all hidden layers
        # assume they come in L x state_size

        # gates
        init_weights = np.random.rand(self.num_layers,
                                      self.state_size * self.num_layers)
        gate_weights = tf.Variable(init_weights, dtype=tf.float32)
        col_inputs = tf.reshape(self.hidden_states,
                                (self.num_layers * self.state_size, 1))
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
        b3 = tf.Variable(np.random.rand(self.output_size, 1), dtype=tf.float32)
        w3 = tf.Variable(
            np.random.rand(self.output_size, hidden_size), dtype=tf.float32)
        prediction = tf.nn.softmax(tf.matmul(w3, l2) + b3)
        return tf.reshape(prediction, (self.output_size, 1))


    def run(self, signal, epochs=100):
        session = tf.Session()
        init = tf.global_variables_initializer()
        session.run(init)

        for e in range(epochs):

            # for first run
            last_run = [(np.random.rand(self.state_size, 1), np.random.rand(
                self.state_size, 1), 1.) for _ in self.layers]
            current_run = [(np.random.rand(self.state_size, 1), np.random.rand(
                self.state_size, 1), 1.) for _ in self.layers]

            step_size = 1
            batch_start = 0
            batch_end = self.batch_size
            while batch_end <= len(signal):

                for t, s in enumerate(signal[batch_start:batch_end]):

                    for i, l in enumerate(self.layers):
                        # short circut copy operation
                        if last_run[i][2] == 0. and current_run[i -
                                                                1][2] == 0.:
                            current_run[i] = last_run[i]
                            continue

                        placeholders = self._get_placeholders(
                            last_run, current_run, i, s, (e * len(signal)) + t)

                        current_run[i] = session.run(l, placeholders)

                    last_run = current_run

                    hidden_states = np.array([h[1][:, 0] for h in current_run])

                    loss = session.run([self.loss], {
                        self.hidden_states: hidden_states,
                        self.current_output: signal[t + 1].reshape(1, self.output_size)
                    })

                    print(loss)
                    session.run([self.train], {
                        self.current_loss: loss,
                    })


                batch_start += step_size
                batch_end = batch_start + self.batch_size

    def calculate_loss(self):
        return tf.losses.softmax_cross_entropy(self.current_output,
                                               self.prediction)

    def minimize_loss(self):
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        return optimizer.minimize(self.current_loss)

    def _get_placeholders(self, last_run, current_run, i, s, iteration):
        # for top layer, the hidden state from above is zero
        if i == len(self.layers) - 1:
            ha = np.zeros((self.state_size, 1))
        else:
            ha = last_run[i + 1][1]

        # for bottom layer, input is signal
        if i == 0:
            hb = s.reshape(-1, 1)
        else:
            hb = current_run[i - 1][1]

        placeholders = {
            self.c: last_run[i][0],
            self.h: last_run[i][1],
            self.z: last_run[i][2],
            self.h_below: hb,
            self.h_above: ha,
            self.z_below: 1 if i == 0 else current_run[i - 1][2],
            self.iteration: iteration,
        }
        return placeholders


def text():
    text = load_text()
    text = text.replace('\n', ' ')
    text = re.sub(' +', ' ', text)
    return one_hot_encode(text)


def load_text():
    with open('text.txt', 'r') as f:
        text = f.read()
    return text


def one_hot_encode(text):
    out = np.zeros((len(text), 29))

    def get_index(char):
        answers = {',': 26, '.': 27}

        if char in answers:
            return answers[char]

        try:
            return ascii_lowercase.index(char)
        except:
            return 28

    for i, t in enumerate(text):
        out[i, get_index(t)] = 1

    return out

if __name__ == '__main__':
    y = text()
    m = hmlstm()
    m.run(y)
