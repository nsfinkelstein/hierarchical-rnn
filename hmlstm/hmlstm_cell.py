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
        c = tf.zeros([batch_size, self._num_units])
        h = tf.zeros([batch_size, self._num_units])
        z = tf.zeros([batch_size])
        return HMLSTMState(c=c, h=h, z=z)

    def call(self, inputs, state):
        """
        Hierarchical multi-scale long short-term memory cell (HMLSTM)

        inputs: [B, hb_l + 1 + ha_l]
        state: (c=[B, h_l], h=[B, h_l], z=[B, 1])

        output: [B, h_l + 1]
        new_state: (c=[B, h_l], h=[B, h_l], z=[B, 1])
        """
        c = state.c                 # [B, h_l]
        h = state.h                 # [B, h_l]
        z = state.z                 # [B, 1]

        in_splits = tf.constant([self._h_below_size, 1, self._h_above_size])

        hb, zb, ha = array_ops.split(
            value=inputs,
            num_or_size_splits=in_splits,
            axis=1,
            name='split')           # [B, hb_l], [B, 1], [B, ha_l]

        s_recurrent = h             # [B, h_l]

        expanded_z = z              # [B, 1]
        s_above = tf.multiply(expanded_z, ha)   # [B, ha_l]
        s_below = tf.multiply(zb, hb)           # [B, hb_l]

        length = 4 * self._num_units + 1
        states = [s_recurrent, s_above, s_below]

        bias_init = tf.constant_initializer(-1e5, dtype=tf.float32)
        # [B, 4 * h_l + 1]
        concat = rnn_cell_impl._linear(states, length, bias=False,
                                       bias_initializer=bias_init)

        gate_splits = tf.constant(
            ([self._num_units] * 4) + [1], dtype=tf.int32)

        i, g, f, o, z_tilde = array_ops.split(
            value=concat, num_or_size_splits=gate_splits, axis=1)

        i = tf.sigmoid(i)           # [B, h_l]
        g = tf.tanh(g)              # [B, h_l]
        f = tf.sigmoid(f)           # [B, h_l]
        o = tf.sigmoid(o)           # [B, h_l]

        new_c = self.calculate_new_cell_state(c, g, i, f, z, zb)
        new_h = self.calculate_new_hidden_state(h, o, new_c, z, zb)
        new_z = tf.expand_dims(self.calculate_new_indicator(z_tilde), -1)

        output = array_ops.concat((new_h, new_z), axis=1)   # [B, h_l + 1]
        new_state = HMLSTMState(c=new_c, h=new_h, z=new_z)

        return output, new_state

    def calculate_new_cell_state(self, c, g, i, f, z, zb):
        '''
        update c and h according to correct operations

        c, g, i, f: [B, h_l]
        z, zb: [B, 1]

        new_c: [B, h_l]
        '''
        z = tf.squeeze(z, axis=[1])                           # [B]
        zb = tf.squeeze(zb, axis=[1])                         # [B]
        new_c = tf.where(
            tf.equal(z, tf.constant(1., dtype=tf.float32)),   # [B]
            tf.multiply(i, g, name='c'),                      # [B, h_l], flush
            tf.where(
                tf.equal(zb, tf.constant(0., dtype=tf.float32)),    # [B]
                tf.identity(c),                               # [B, h_l], copy
                tf.add(tf.multiply(f, c), tf.multiply(i, g))  # [B, h_l], update
            )
        )
        return new_c  # [B, h_l]

    def calculate_new_hidden_state(self, h, o, new_c, z, zb):
        '''
        h, o, new_c: [B, h_l]
        z, zb: [B, 1]

        new_h: [B, h_l]
        '''
        z = tf.squeeze(z, axis=[1])             # [B]
        zb = tf.squeeze(zb, axis=[1])           # [B]
        new_h = tf.where(
            tf.logical_and(
                tf.equal(z, tf.constant(0., dtype=tf.float32)),
                tf.equal(zb, tf.constant(0., dtype=tf.float32))
            ),                                  # [B]
            tf.identity(h),                     # [B, h_l], if copy
            tf.multiply(o, tf.tanh(new_c))      # [B, h_l], otherwise
        )
        return new_h                            # [B, h_l]

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

