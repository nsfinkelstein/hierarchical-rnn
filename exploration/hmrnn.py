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
                 output_size=29,
                 embedding_size=100,
                 out_hidden_size=200):

        # model configuration values
        self.state_size = state_size
        self.batch_size = batch_size
        self.step_size = step_size
        self.output_size = output_size
        self.num_layers = layers
        self.embedding_size = embedding_size
        self.out_hidden_size = 200

        # placeholders
        out_shape = (batch_size, output_size)
        in_shape = (batch_size, output_size, 1)
        self.batch_output = tf.placeholder(tf.int32, shape=out_shape)
        self.batch_input = tf.placeholder(tf.float32, shape=in_shape)
        self.epoch = tf.placeholder(tf.float32, shape=())

        # output module variables
        init_gate_weights = np.random.rand(layers, state_size * layers)
        init_emb_weights = np.random.rand(state_size, embedding_size)
        self.gate_weights = tf.Variable(init_gate_weights, dtype=tf.float32)
        self.embed_weights = tf.Variable(init_emb_weights, dtype=tf.float32)

        def init_bias():
            return np.random.rand(out_hidden_size, 1)

        self.b1 = tf.Variable(init_bias(), dtype=tf.float32)
        self.b2 = tf.Variable(init_bias(), dtype=tf.float32)
        self.b3 = tf.Variable(np.random.rand(output_size, 1), dtype=tf.float32)
        init_w1 = np.random.rand(out_hidden_size, embedding_size)
        self.w1 = tf.Variable(init_w1, dtype=tf.float32)
        init_w2 = np.random.rand(out_hidden_size, out_hidden_size)
        self.w2 = tf.Variable(init_w2, dtype=tf.float32)
        init_w3 = np.random.rand(output_size, out_hidden_size)
        self.w3 = tf.Variable(init_w3, dtype=tf.float32)

        # hmlstm layers variables
        weight_size = (state_size * 4) + 1
        self.b = tf.Variable(np.zeros((weight_size, 1)), dtype=tf.float32)

        def initialize_weights():
            return np.random.rand(weight_size, self.state_size)

        self.w = tf.Variable(initialize_weights(), dtype=tf.float32)
        self.wa = tf.Variable(initialize_weights(), dtype=tf.float32)
        self.wb = tf.Variable(initialize_weights(), dtype=tf.float32)

    def hmlstm_layer(self, c, h, z, h_below, h_above, z_below):
        # calculate LSTM-like gates
        s_recurrent = tf.matmul(self.w, h)
        s_top = tf.multiply(z, tf.matmul(self.wa, h_above))
        s_bottom = tf.multiply(z_below, tf.matmul(self.wb, h_below))
        joint_input = tf.add_n((s_recurrent, s_top, s_bottom, self.b))

        f = tf.sigmoid(joint_input[:self.state_size])
        i = tf.sigmoid(joint_input[self.state_size:2 * self.state_size])
        o = tf.sigmoid(joint_input[2 * self.state_size:3 * self.state_size])
        g = tf.tanh(joint_input[3 * self.state_size:4 * self.state_size])

        # these are the three possible operations
        def copy():
            return (c, h)

        def update():
            nc = tf.add(tf.multiply(f, c), tf.multiply(i, g))
            nh = tf.multiply(o, tf.tanh(nc))
            return nc, nh

        def flush():
            nc = tf.multiply(i, g)
            nh = tf.multiply(o, tf.tanh(nc))
            return nc, nh

        # calculate new cell and hidden states
        new_c, new_h = tf.case(
            [
                (tf.equal(z, tf.constant(1, dtype=tf.float32)), flush),
                (tf.logical_and(
                    tf.equal(z, tf.constant(0, dtype=tf.float32)),
                    tf.equal(z_below, tf.constant(0, dtype=tf.float32))),
                 copy),
                (tf.logical_and(
                    tf.equal(z, tf.constant(0, dtype=tf.float32)),
                    tf.equal(z_below, tf.constant(1, dtype=tf.float32))),
                 update),
            ],
            default=update,
            exclusive=True)

        # use slope annealing trick
        slope_multiplier = tf.maximum(.02 + (self.epoch / 1e5),
                                      tf.constant(5.))
        z_tilde = tf.sigmoid(joint_input[-1:] * slope_multiplier)

        # replace gradient calculation - use straight-through estimator
        # see: https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
        graph = tf.get_default_graph()
        with ops.name_scope('BinaryRound') as name:
            with graph.gradient_override_map({'Round': 'Identity'}):
                new_z = tf.round(z_tilde, name=name)

        return new_c, new_h, tf.squeeze(new_z)

    def output_module(self, hidden_states):
        # inputs are concatenated output from all hidden layers
        # assume they come in L x state_size

        # gates
        col_inputs = tf.reshape(hidden_states,
                                (self.num_layers * self.state_size, 1))

        gates = tf.sigmoid(tf.matmul(self.gate_weights, col_inputs))
        gated = tf.multiply(gates, hidden_states)

        # embedding
        prod = tf.matmul(gated, self.embedding_weights)
        embedding = tf.nn.relu(tf.reduce_sum(prod, axis=0))
        col_embedding = tf.reshape(embedding, (self.embedding_size, 1))

        # feed forward network
        # first layer
        l1 = tf.nn.tanh(tf.matmul(self.w1, col_embedding) + self.b1)

        # second layer
        l2 = tf.nn.tanh(tf.matmul(self.w2, l1) + self.b2)

        # the loss function used below
        # sparse_softmax_cross_entropy_with_logits
        # calls softmax, so we just pass the linear result through here
        prediction = tf.matmul(self.w3, l2) + self.b3
        return tf.reshape(prediction, (self.output_size, 1))

    def full_stack(self):
        states = [[0] * self.num_layers] * self.batch_size

        # results are stored in the order: c, h, z
        total_loss = tf.constant(0.0)
        for t in range(self.batch_size):
            for l in range(self.num_layers):
                args = self._get_hmlstm_args(t, l, states)
                states[t][l] = self.hmlstm_layer(*args)

            hidden_states = tf.stack([h for c, h, z in states[t]])
            prediction = self.output_module(hidden_states)
            total_loss = tf.add(total_loss,
                                tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits=prediction,
                                    labels=self.batch_output[t]))
        return total_loss

    def _get_hmlstm_args(self, t, l, states):
        if l == 0:
            z_below = 1.
            h_below = self.batch_input[t]
        else:
            z_below = states[t][l - 1][2]
            h_below = states[t][l - 1][1]

        if t == 0:
            c = tf.zeros([self.state_size, 1])
            h = tf.zeros([self.state_size, 1])
            h_above = tf.zeros([self.state_size, 1])
            z = 1.
        else:
            c = states[t - 1][l][0]
            h = states[t - 1][l][1]
            z = states[t - 1][l][2]

            if l != self.num_layers - 1:
                h_above = states[t - 1][l + 1][1]
            else:
                h_above = tf.zeros([self.state_size, 1])

        return c, h, z, h_below, h_above, z_below

    def run(self, signal, epochs=100):
        session = tf.Session()
        init = tf.global_variables_initializer()
        session.run(init)

        for epoch in range(epochs):
            step_size = 1
            batch_start = 0
            batch_end = self.batch_size
            while batch_end < len(signal):
                batch_in = signal[batch_start:batch_end]
                batch_out = signal[batch_start + 1:batch_end + 1]

                loss = self.full_stack()
                train = self.minimize_loss(loss)
                _loss, _ = session.run(
                    [loss, train],
                    feed_dict={
                        self.batch_input:
                        batch_in.reshape(self.batch_size, -1, 1),
                        self.batch_output:
                        batch_out,
                        self.epoch:
                        epoch,
                    })
                print(_loss)

                batch_start += step_size
                batch_end = batch_start + self.batch_size

    def minimize_loss(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        return optimizer.minimize(loss)


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
