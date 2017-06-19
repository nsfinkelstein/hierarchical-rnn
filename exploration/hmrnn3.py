from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf
import collections

HMLSTMState = collections.namedtuple('HMLSTMCellState', ('c', 'h', 'z'))


class HMLSTMCell(rnn_cell_impl.RNNCell):
    def __init__(self, num_units):
        self._num_units = num_units

    @property
    def state_size(self):
        # the state is c, h, and z
        return (self._num_units, self._num_units, 1)

    @property
    def output_size(self):
        # outputs h and z
        return self._num_units + 1

    def zero_state(self, batch_size, dtype):
        return HMLSTMState(
            c=tf.zeros([self._num_units]),
            h=tf.zeros([self._num_units]),
            z=tf.zeros(()))

    def __call__(self, inputs, state):
        """Hierarchical multi-scale long short-term memory cell (HMLSTM)"""
        c, h, z = state

        in_splits = tf.constant(([self._num_units] * 2) + [1])
        print(inputs)
        ha, hb, zb = array_ops.split(
            value=inputs, num_or_size_splits=in_splits, axis=2)

        s_recurrent = h
        s_above = tf.multiply(z, ha)
        s_below = tf.multiply(zb, hb)

        length = 4 * self._num_units + 1
        states = [s_recurrent, s_above, s_below]
        concat = rnn_cell_impl._linear(states, length, bias=True)

        gate_splits = tf.constant(
            ([self._num_inputs] * 4) + [1], dtype=tf.int32)

        i, g, f, o, z_tilde = array_ops.split(
            value=concat, num_or_size_splits=gate_splits, axis=1)

        new_c = self.calculate_new_cell_state(c, g, i, f, z, zb)
        new_h = self.calculate_new_cell_state(h, o, new_c, z, zb)
        new_z = self.calculate_new_indicator(z_tilde)

        output = array_ops.concat((new_h, new_z), axis=1)
        new_state = HMLSTMState(new_c, new_h, new_z)
        return output, new_state

    def calculate_new_cell_state(self, c, g, i, f, z, zb):
        # update c and h according to correct operations
        def copy_c():
            return c

        def update_c():
            return tf.add(tf.multiply(f, c), tf.multiply(i, g))

        def flush_c():
            return tf.multiply(i, g, name='c')

        new_c = tf.case(
            [
                (tf.equal(z, tf.constant(1., dtype=tf.float32)), flush_c),
                (tf.logical_and(
                    tf.equal(z, tf.constant(0., dtype=tf.float32)),
                    tf.equal(zb, tf.constant(0., dtype=tf.float32))), copy_c),
                (tf.logical_and(
                    tf.equal(z, tf.constant(0., dtype=tf.float32)),
                    tf.equal(zb, tf.constant(1., dtype=tf.float32))),
                 update_c),
            ],
            default=update_c,
            exclusive=True)
        return new_c

    def calculate_new_hidden_state(self, h, o, new_c, z, zb):
        def copy_h():
            return h

        def update_h():
            return tf.multiply(o, tf.tanh(new_c))

        new_h = tf.cond(
            tf.logical_and(
                tf.equal(z, tf.constant(0., dtype=tf.float32)),
                tf.equal(zb, tf.constant(0., dtype=tf.float32))), copy_h,
            update_h)
        return new_h

    def calculate_new_indicator(self, z_tilde):
        # use slope annealing trick
        slope_multiplier = 1  # tf.maximum(tf.constant(.02) + self.epoch, tf.constant(5.))

        # replace gradient calculation - use straight-through estimator
        # see: https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
        graph = tf.get_default_graph()
        with ops.name_scope('BinaryRound') as name:
            with graph.gradient_override_map({'Round': 'Identity'}):
                new_z = tf.round(z_tilde, name=name)

        return tf.squeeze(new_z)


class MultiHMLSTMCell(rnn_cell_impl.RNNCell):
    """HMLSTM cell composed squentially of individual HMLSTM cells."""

    def __init__(self, cells):
        self._cells = cells

    def zero_state(self, batch_size, dtype):
        name = type(self).__name__ + 'ZeroState'
        with ops.name_scope(name, values=[batch_size]):
            return tuple(
                cell.zero_state(batch_size, dtype) for cell in self._cells)

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)

    @property
    def output_size(self):
        return self._cells[-1].output_size

    def __call__(self, inputs, state):
        """Run this multi-layer cell on inputs, starting from state."""
        raw_inp = inputs[:, :, :-sum(c._num_units for c in self._cells)]

        # split out the part of the input that stores values of ha
        raw_h_aboves = inputs[:, :, -sum(c._num_units for c in self._cells):]
        h_aboves = array_ops.split(
            value=raw_h_aboves, num_or_size_splits=len(self._cells), axis=2)

        z_below = tf.ones([tf.shape(inputs)[0], 1, 1])

        new_states = []
        for i, cell in enumerate(self._cells):
            with vs.variable_scope("cell_%d" % i):
                cur_state = state[i]

                cur_inp = array_ops.concat([raw_inp, h_aboves[i], z_below], axis=2)
                raw_inp, new_state = cell(cur_inp, cur_state)
                new_states.append(new_state)

            new_states = tuple(new_states)

        hidden_states = [ns['h'] for ns in new_states]

        return hidden_states, tuple(new_states)


class MultiHMLSTMNetwork(object):
    def __init__(self, batch_size, num_layers, truncate_len, num_units):
        self._batch_size = batch_size
        self._num_layers = num_layers
        self._truncate_len = truncate_len
        self._num_units = num_units  # the length of c and h

        batch_shape = (batch_size, truncate_len, num_units)
        self.batch_in = tf.placeholder(tf.float32, shape=batch_shape,
                                       name='batch_in')

        self.batch_out = tf.placeholder(tf.float32, shape=batch_shape,
                                        name='batch_out')

        self.train, self.loss = self.create_network(self.create_output_module)

    def gate_input(self, hidden_states):
        # gate the incoming hidden states
        with vs.variable_scope('gates'):
            gates = []
            for l in range(self._num_layers):
                weights = vs.get_variable(
                    'gate_%s' % l, [self._num_units * 4, 1], dtype=tf.float32)
                gates.append(tf.sigmoid(tf.matmul(weights, hidden_states)))

            split = array_ops.split(
                value=hidden_states, num_or_size_splits=self._num_layers)
            gated_list = []
            for gate, hidden_state in zip(gates, split):
                gated_list.append(tf.multiply(gate, hidden_state))

            gated_input = tf.concat(gated_list, axis=1)
        return gated_input

    def create_output_module(self, gated_input):
        embed_size = 100
        out_hidden_size = 100
        with vs.variable_scope('output_module'):

            in_size = self._num_layers * self._num_units
            embed_shape = [in_size, embed_size]
            embed_weights = vs.get_variable(
                'embed_weights', embed_shape, dtype=tf.float32)

            b1 = vs.get_variable('b1', [out_hidden_size, 1], dtype=tf.float32)
            b2 = vs.get_variable('b2', [out_hidden_size, 1], dtype=tf.float32)
            b3 = vs.get_variable('b3', [self._num_units, 1], dtype=tf.float32)
            w1 = vs.get_variable(
                'w1', [embed_size, out_hidden_size], dtype=tf.float32)
            w2 = vs.get_variable(
                'w2', [out_hidden_size, out_hidden_size], dtype=tf.float32)
            w3 = vs.get_variable(
                'w3', [out_hidden_size, self._num_units], dtype=tf.float32)

            # embedding
            prod = tf.matmul(gated_input, embed_weights)
            embedding = tf.nn.relu(tf.reduce_sum(prod, axis=1))

            # feed forward network
            # first layer
            l1 = tf.nn.tanh(tf.matmul(w1, embedding) + b1)

            # second layer
            l2 = tf.nn.tanh(tf.matmul(w2, l1) + b2)

            # the loss function used below
            # sparse_softmax_cross_entropy_with_logits
            prediction = tf.matmul(w3, l2) + b3

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.batch_out, logits=tf.stack(prediction))
        return loss

    def create_network(self, output_module):
        hmlstm_cell = HMLSTMCell(self._num_units)
        hmlstm = MultiHMLSTMCell([hmlstm_cell] * self._num_layers)

        state = hmlstm.zero_state(self._batch_size, tf.float32)
        h_aboves = tf.zeros([self._batch_size, 1, (self._num_layers * self._num_units)])
        loss = 0.0
        for i in range(self._truncate_len):
            inputs = array_ops.concat((self.batch_in[:, i:(i+1)], h_aboves), axis=2)

            hidden_states, state = hmlstm(inputs, state)
            concated_hs = array_ops.concat(hidden_states[1:], axis=0)
            h_aboves[:, 0, :-self._num_units] = concated_hs
            loss += output_module(concated_hs)

        train = tf.train.AdamOptimizer(1e-4).minimize(loss)
        return train, loss

    def run(self, batches_in, batches_out):
        for batch_in, batch_out in zip(batches_in, batches_out):
            with tf.Session() as sess:
                _train, _loss = sess.run(self.train, self.loss, {
                    self.batch_in: batch_in,
                    self.batch_out: batch_out,
                })
            print(_loss)


from string import ascii_lowercase
import re
import numpy as np

batch_size = 10
truncate_len = 100

def text():
    text = load_text()
    text = text.replace('\n', ' ')
    signals = re.sub(' +', ' ', text)

    return [(one_hot_encode(intext), one_hot_encode(outtext))
            for intext, outtext in signals]

def load_text():
    with open('text.txt', 'r') as f:
        text = f.read()

    signals = []
    for i in range(0, 50 * 100, 50):
        intext = text[i:i + truncate_len]
        outtext = text[i + 1:i + truncate_len + 1]
        signals.append((intext, outtext))

    return signals

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

def run_everything():
    network = MultiHMLSTMNetwork(batch_size, 3, truncate_len, 29)
    y = text()
    batches_in = []
    batches_out = []
    for batch_number in range(10):
        start = batch_number * batch_size
        end = (1 + batch_number) * batch_size
        batches_in.append([i for i, _ in y[start: end]])
        batches_out.append([o for _, o in y[start:end]])
    network.run(batches_in, batches_out)

if __name__ == '__main__':
    run_everything()
