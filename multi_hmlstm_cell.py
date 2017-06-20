from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf


class MultiHMLSTMCell(rnn_cell_impl.RNNCell):
    """HMLSTM cell composed squentially of individual HMLSTM cells."""

    def __init__(self, cells):
        super(MultiHMLSTMCell, self).__init__(_reuse=None)
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

    def call(self, inputs, state):
        """Run this multi-layer cell on inputs, starting from state."""
        raw_inp = inputs[:, :, :-sum(c._num_units for c in self._cells)]

        # split out the part of the input that stores values of ha
        raw_h_aboves = inputs[:, :, -sum(c._num_units for c in self._cells):]
        h_aboves = array_ops.split(
            value=raw_h_aboves, num_or_size_splits=len(self._cells), axis=2)

        z_below = tf.ones([tf.shape(inputs)[0], 1, 1])

        raw_inp = array_ops.concat([raw_inp, z_below], axis=2)

        new_states = []
        for i, cell in enumerate(self._cells):
            with vs.variable_scope("cell_%d" % i):
                cur_state = state[i]

                print(i)
                cur_inp = array_ops.concat(
                    [raw_inp, h_aboves[i]], axis=2, name='input_to_cell')
                raw_inp, new_state = cell(cur_inp, cur_state)
                raw_inp = tf.expand_dims(raw_inp, 1)
                new_states.append(new_state)

        hidden_states = [ns[1] for ns in new_states]
        return hidden_states, tuple(new_states)
