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

    def zero_state(self):
        return HMLSTMCellState(c=)

    def __call__(self, inputs, state):
        """Hierarchical multi-scale long short-term memory cell (HMLSTM)"""
        c, h, z = state

        in_splits = tf.constant(([self._num_units] * 2) + [1])
        ha, hb, zb = array_ops.split(
            value=inputs, num_or_size_splits=in_splits, axis=1)

        s_recurrent = h
        s_above = tf.multiply(z, ha)
        s_below = tf.multiply(zb, hb)

        length = 4 * self._num_units + 1
        states = [s_recurrent, s_above, s_below]
        concat = rnn_cell_impl._linear(states, length, bias=True)

        gate_splits = tf.constant(([self._num_inputs] * 4) + [1],
                                  dtype=tf.int32)

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
            return tuple(cell.zero_state(batch_size, dtype)
                         for cell in self._cells)

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)

    @property
    def output_size(self):
        return self._cells[-1].output_size

    def __call__(self, inputs, state):
        """Run this multi-layer cell on inputs, starting from state."""
        raw_inp = inputs[:-sum(c.state_size for c in self._cells)]

        # split out the part of the input that stores values of ha
        raw_h_aboves = inputs[-self._cells[0].state_size * len(self._cells):]
        h_aboves = array_ops.split(value=raw_h_aboves,
                                   num_or_size_splits=len(self._cell))

        new_states = []
        for i, cell in enumerate(self._cells):
            with vs.variable_scope("cell_%d" % i):
                cur_state = state[i]

                cur_inp = array_ops.concat([raw_inp, h_aboves[i]], axis=1)
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

        output_module = self.create_output_module()
        network = self.create_network(output_module)
        self.batch_in = tf.placeholder(tf.float32,
                                  shape=(batch_size, truncate_len),
                                  name='batch_in')

        self.batch_out = tf.placeholder(tf.float32,
                                   shape=(batch_size, truncate_len),
                                   name='batch_out')


    def create_output_module(self):
        pass

    def create_network(self, output_module):
        hmlstm_cell = HMLSTMCell(self._num_units)
        hmlstm = MultiHMLSTMCell([hmlstm_cell] * num_layers)

        state = hmlstm.zero_state(batch_size, tf.float32)
        h_aboves = tf.zeros([batch_size, self._num_units])
        loss = 0.0
        for i in range(truncate_len):
            inputs = array_ops.concat((batch_in[:, i], h_aboves), axis=1)

            hidden_states, state = hmlstm(inputs, state)
            concated_hs = array_ops.concat(hidden_states[1:], axis=0)

            # TODO: write output module
            prediction = output_module(concated_hs)

            h_aboves[:-self._num_units] = concated_hs

            loss += tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=batch_out[:, i], logits=predictions
            )

        return loss

    def run(self, batch_in, batch_out):
        _loss = self.network()




.

        
