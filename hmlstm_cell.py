from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell_impl
import tensorflow as tf
import collections


HMLSTMState = collections.namedtuple('HMLSTMCellState', ('c', 'h', 'z'))


class HMLSTMCell(rnn_cell_impl.RNNCell):
    def __init__(self, num_units, batch_size):
        super(HMLSTMCell, self).__init__(_reuse=None)
        self._num_units = num_units
        self._batch_size = batch_size

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
            c=tf.zeros([batch_size, self._num_units]),
            h=tf.zeros([batch_size, self._num_units]),
            z=tf.zeros([batch_size]))

    def call(self, inputs, state):
        """Hierarchical multi-scale long short-term memory cell (HMLSTM)"""
        c, h, z = state

        in_splits = tf.constant([self._num_units, 1, self._num_units])
        hb, zb, ha = array_ops.split(
            value=inputs,
            num_or_size_splits=in_splits,
            axis=2,
            name='split_input')

        s_recurrent = h
        expanded_z = tf.expand_dims(tf.expand_dims(z, -1), -1)
        s_above = tf.squeeze(tf.multiply(expanded_z, ha), axis=1)
        s_below = tf.squeeze(tf.multiply(zb, hb), axis=1)

        length = 4 * self._num_units + 1
        states = [s_recurrent, s_above, s_below]
        concat = rnn_cell_impl._linear(states, length, bias=True)

        gate_splits = tf.constant(
            ([self._num_units] * 4) + [1], dtype=tf.int32)

        i, g, f, o, z_tilde = array_ops.split(
            value=concat, num_or_size_splits=gate_splits, axis=1)

        new_c = self.calculate_new_cell_state(c, g, i, f, z, zb)
        new_h = self.calculate_new_hidden_state(h, o, new_c, z, zb)
        new_z = self.calculate_new_indicator(z_tilde)

        output = array_ops.concat((new_h, tf.expand_dims(new_z, -1)), axis=1)
        new_state = HMLSTMState(new_c, new_h, new_z)

        return output, new_state

    def calculate_new_cell_state(self, c, g, i, f, z, zb):
        # update c and h according to correct operations
        # must do each batch independently
        new_c = [0] * self._batch_size
        for b in range(self._batch_size):

            def copy_c():
                return c[b]

            def update_c():
                return tf.add(tf.multiply(f[b], c[b]), tf.multiply(i[b], g[b]))

            def flush_c():
                return tf.multiply(i[b], g[b], name='c')

            new_c[b] = tf.case(
                [
                    (tf.equal(
                        tf.squeeze(z[b]), tf.constant(1., dtype=tf.float32)),
                     flush_c),
                    (tf.logical_and(
                        tf.equal(
                            tf.squeeze(z[b]), tf.constant(
                                0., dtype=tf.float32)),
                        tf.equal(
                            tf.squeeze(zb[b]),
                            tf.constant(0., dtype=tf.float32))), copy_c),
                    (tf.logical_and(
                        tf.equal(
                            tf.squeeze(z[b]), tf.constant(
                                0., dtype=tf.float32)),
                        tf.equal(
                            tf.squeeze(zb[b]),
                            tf.constant(1., dtype=tf.float32))), update_c),
                ],
                default=update_c,
                exclusive=True)

        return tf.stack(new_c, axis=0)

    def calculate_new_hidden_state(self, h, o, new_c, z, zb):
        new_h = [0] * self._batch_size
        for b in range(self._batch_size):

            def copy_h():
                return h[b]

            def update_h():
                return tf.multiply(o[b], tf.tanh(new_c[b]))

            new_h[b] = tf.cond(
                tf.logical_and(
                    tf.equal(
                        tf.squeeze(z[b]), tf.constant(0., dtype=tf.float32)),
                    tf.equal(
                        tf.squeeze(zb[b]), tf.constant(0., dtype=tf.float32))),
                copy_h, update_h)
        return tf.stack(new_h, axis=0)

    def calculate_new_indicator(self, z_tilde):
        for i in range(batch_size):
            tf.summary.scalar('z_tilde' + str(i), tf.squeeze(z_tilde[i]))
        # use slope annealing trick
        slope_multiplier = 1  # tf.maximum(tf.constant(.02) + self.epoch, tf.constant(5.))
        sigmoided = tf.sigmoid(z_tilde) * slope_multiplier

        # replace gradient calculation - use straight-through estimator
        # see: https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
        graph = tf.get_default_graph()
        with ops.name_scope('BinaryRound') as name:
            with graph.gradient_override_map({'Round': 'Identity'}):
                new_z = tf.round(sigmoided, name=name)

        return tf.squeeze(new_z, axis=1)


