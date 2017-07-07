from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell_impl
import tensorflow as tf
import collections


HMLSTMState = collections.namedtuple('HMLSTMCellState', ['c', 'h', 'z'])


class HMLSTMCell(rnn_cell_impl.RNNCell):
    def __init__(self, num_units, batch_size, h_below_size, h_above_size,
                 reuse):
        super(HMLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._h_below_size = h_below_size
        self._h_above_size = h_above_size
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
        c = tf.zeros([batch_size, self._num_units], name='first_c_xxx')
        h = tf.zeros([batch_size, self._num_units], name='first_h_xxx')
        z = tf.zeros([batch_size], name='first_z_xxx')
        return HMLSTMState(c=c, h=h, z=z)

    def call(self, inputs, state):
        """Hierarchical multi-scale long short-term memory cell (HMLSTM)"""
        c = state.c
        h = state.h
        z = state.z

        in_splits = tf.constant([self._h_below_size, 1, self._h_above_size])

        hb, zb, ha = array_ops.split(
            value=inputs,
            num_or_size_splits=in_splits,
            axis=1,
            name='split')

        s_recurrent = h

        expanded_z = z
        s_above = tf.multiply(expanded_z, ha)
        s_below = tf.multiply(zb, hb)

        length = 4 * self._num_units + 1
        states = [s_recurrent, s_above, s_below]

        bias_init = tf.constant_initializer(-1e5, dtype=tf.float32)
        concat = rnn_cell_impl._linear(states, length, bias=False,
                                       bias_initializer=bias_init)

        gate_splits = tf.constant(
            ([self._num_units] * 4) + [1], dtype=tf.int32)

        i, g, f, o, z_tilde = array_ops.split(
            value=concat, num_or_size_splits=gate_splits, axis=1)

        i = tf.sigmoid(i)
        g = tf.tanh(g)
        f = tf.sigmoid(f)
        o = tf.sigmoid(o)

        new_c = self.calculate_new_cell_state(c, g, i, f, z, zb)
        new_h = self.calculate_new_hidden_state(h, o, new_c, z, zb)
        new_z = tf.expand_dims(self.calculate_new_indicator(z_tilde), -1)

        output = array_ops.concat((new_h, new_z), axis=1)
        new_state = HMLSTMState(c=new_c, h=new_h, z=new_z)

        return output, new_state

    def calculate_new_cell_state(self, c, g, i, f, z, zb):
        # update c and h according to correct operations
        # must do each batch independently
        new_c = [0] * self._batch_size
        for b in range(self._batch_size):

            def copy_c():
                return tf.identity(c[b])

            def update_c():
                return tf.add(tf.multiply(f[b], c[b]), tf.multiply(i[b], g[b]))

            def flush_c():
                return tf.multiply(i[b], g[b], name='c')

            def default_c():
                return tf.zeros_like(flush_c())

            new_c[b] = tf.case(
                {
                    tf.equal(
                        tf.squeeze(z[b]), tf.constant(1., dtype=tf.float32),
                        name='flush_c_xxx'
                    ): flush_c,
                    tf.logical_and(
                        tf.equal(
                            tf.squeeze(z[b], name='squeeze_z_xxx'),
                            tf.constant(0., dtype=tf.float32)),
                        tf.equal(
                            tf.squeeze(zb[b], name='squeeze_zb_xxx'),
                            tf.constant(0., dtype=tf.float32)),
                        name='copy_c_xxx'
                    ): copy_c,
                    tf.logical_and(
                        tf.equal(
                            tf.squeeze(z[b]),
                            tf.constant(0., dtype=tf.float32)),
                        tf.equal(
                            tf.squeeze(zb[b]),
                            tf.constant(1., dtype=tf.float32)),
                        name='update_c_xxx'
                    ): update_c,
                },
                default=default_c,
                exclusive=True)

        return tf.stack(new_c, axis=0)

    def calculate_new_hidden_state(self, h, o, new_c, z, zb):
        new_h = [0] * self._batch_size
        for b in range(self._batch_size):

            def copy_h():
                return tf.identity(h[b])

            def update_h():
                return tf.multiply(o[b], tf.tanh(new_c[b]))

            new_h[b] = tf.cond(
                tf.logical_and(
                    tf.equal(
                        tf.squeeze(z[b]),
                        tf.constant(0., dtype=tf.float32)),
                    tf.equal(
                        tf.squeeze(zb[b]),
                        tf.constant(0., dtype=tf.float32))),
                copy_h, update_h)

        return tf.stack(new_h, axis=0)

    def calculate_new_indicator(self, z_tilde):
        # use slope annealing trick
        slope_multiplier = 1  # NOTE: Change this for some tasks
        sigmoided = tf.sigmoid(z_tilde * slope_multiplier)

        # replace gradient calculation - use straight-through estimator
        # see: https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
        graph = tf.get_default_graph()
        with ops.name_scope('BinaryRound') as name:
            with graph.gradient_override_map({'Round': 'Identity'}):
                new_z = tf.round(sigmoided, name=name)

        return tf.squeeze(new_z, axis=1)

