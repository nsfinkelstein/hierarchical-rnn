from hmlstm_cell import HMLSTMCell, HMLSTMState
from multi_hmlstm_cell import MultiHMLSTMCell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf
import numpy as np


class HMLSTMNetwork(object):
    def __init__(self,
                 input_size=29,
                 output_size=1,
                 num_layers=3,
                 hidden_state_sizes=50,
                 out_hidden_size=100,
                 embed_size=100,
                 task='classification'):

        self._out_hidden_size = out_hidden_size
        self._embed_size = embed_size
        self._num_layers = num_layers
        self._input_size = input_size
        self._session = None
        self._graph = dict()

        if type(hidden_state_sizes) == int:
            self._hidden_state_sizes = [hidden_state_sizes] * self._num_layers
        else:
            self._hidden_state_sizes = hidden_state_sizes

        if task == 'classification':
            self._loss_function = tf.nn.softmax_cross_entropy_with_logits
            self._output_size = output_size
        elif task == 'regression':
            self._loss_function = lambda logits, labels: tf.square((logits - labels))
            self._output_size = 1

        batch_shape = (None, None, self._output_size)
        self.batch_in = tf.placeholder(
            tf.float32, shape=batch_shape, name='batch_in')
        self.batch_out = tf.placeholder(
            tf.float32, shape=batch_shape, name='batch_out')

    def load_variables(self, path='./hmlstm_ckpt'):
        if self._session is None:
            self._session = tf.Session()

        saver = tf.train.Saver()
        print('loading variables...')
        saver.restore(self._session, path)

    def save_variables(self, path='./hmlstm_ckpt'):
        saver = tf.train.Saver()
        print('saving variables...')
        saver.save(self._session, path)

    def gate_input(self, hidden_states):
        # gate the incoming hidden states
        with vs.variable_scope('gates_vars'):
            gates = []
            for l in range(self._num_layers):
                weights = vs.get_variable('gate_%d' % l, [sum(self._hidden_state_sizes), 1], dtype=tf.float32)
                gates.append(tf.sigmoid(tf.matmul(hidden_states, weights)))

            split = array_ops.split(
                value=hidden_states,
                num_or_size_splits=self._hidden_state_sizes,
                axis=1)

            gated_list = []
            for gate, hidden_state in zip(gates, split):
                gated_list.append(tf.multiply(gate, hidden_state))

            gated_input = tf.concat(gated_list, axis=1)
        return gated_input

    def embed_input(self, gated_input):
        with vs.variable_scope('embedding_vars'):
            embed_shape = [sum(self._hidden_state_sizes), self._embed_size]
            embed_weights = vs.get_variable('embed_weights', embed_shape, dtype=tf.float32)

            prod = tf.matmul(gated_input, embed_weights)
            embedding = tf.nn.relu(prod)

        return embedding

    def output_module(self, embedding, outcome):
        with vs.variable_scope('output_module_vars'):
            b1 = vs.get_variable('b1', [1, self._out_hidden_size], dtype=tf.float32)
            b2 = vs.get_variable('b2', [1, self._out_hidden_size], dtype=tf.float32)
            b3 = vs.get_variable('b3', [1, self._output_size], dtype=tf.float32)
            w1 = vs.get_variable('w1', [self._embed_size, self._out_hidden_size], dtype=tf.float32)
            w2 = vs.get_variable('w2', [self._out_hidden_size, self._out_hidden_size], dtype=tf.float32)
            w3 = vs.get_variable('w3', [self._out_hidden_size, self._output_size], dtype=tf.float32)

            # feed forward network
            # first layer
            l1 = tf.nn.tanh(tf.matmul(embedding, w1) + b1)

            # second layer
            l2 = tf.nn.tanh(tf.matmul(l1, w2) + b2)

            # the loss function used below
            # softmax_cross_entropy_with_logits
            prediction = tf.add(tf.matmul(l2, w3), b3, name='prediction')

            loss_args = {'logits': prediction, 'labels': outcome}
            loss = self._loss_function(**loss_args)

        return loss, prediction

    def create_multicell(self, batch_size):
        def hmlstm_cell(layer):
            if layer == 0:
                h_below_size = self._input_size
            else:
                h_below_size = self._hidden_state_sizes[layer - 1]

            if layer == self._num_layers - 1:
                # doesn't matter, all zeros, but for convenience with summing
                # so the sum of ha sizes is just sum of hidden states
                h_above_size = self._hidden_state_sizes[0]
            else:
                h_above_size = self._hidden_state_sizes[layer + 1]

            return HMLSTMCell(self._hidden_state_sizes[layer], batch_size,
                              h_below_size, h_above_size)

        hmlstm = MultiHMLSTMCell(
            [hmlstm_cell(l) for l in range(self._num_layers)])

        return hmlstm

    def split_out_cell_states(self, accum):
        splits = []
        for size in self._hidden_state_sizes:
            splits += [size, size, 1]

        split_states = array_ops.split(value=accum,
                                       num_or_size_splits=splits, axis=1)

        cell_states = []
        for l in range(self._num_layers):
            c = split_states[(l * 3)]
            h = split_states[(l * 3) + 1]
            z = split_states[(l * 3) + 2]
            cell_states.append(HMLSTMState(c=c, h=h, z=z))

        return cell_states

    def get_h_aboves(self, hidden_states, batch_size, hmlstm):
        concated_hs = array_ops.concat(hidden_states[1:], axis=1)

        h_above_for_last_layer = tf.zeros(
            [batch_size, hmlstm._cells[-1]._h_above_size], dtype=tf.float32)

        h_aboves = array_ops.concat(
            [concated_hs, h_above_for_last_layer], axis=1)

        return h_aboves

    def network(self, batch_size, truncate_len):
        hmlstm = self.create_multicell(batch_size)

        def scan_rnn(accum, elem):
            # each element is the set of all hidden states from the previous
            # time step
            cell_states = self.split_out_cell_states(accum)

            h_aboves = self.get_h_aboves([cs.h for cs in cell_states],
                                         batch_size, hmlstm)

            hmlstm_in = array_ops.concat((elem, h_aboves), axis=1)
            _, state = hmlstm(hmlstm_in, cell_states)

            concated_states = [array_ops.concat(tuple(s), axis=1) for s in state]
            return array_ops.concat(concated_states, axis=1)

        elem_len = (sum(self._hidden_state_sizes) * 2) + self._num_layers
        initial = tf.zeros([batch_size, elem_len])

        states = tf.scan(scan_rnn, self.batch_in, initial)
        timestep_states = tf.unstack(states, num=truncate_len, axis=0)
        cell_states = [self.split_out_cell_states(s) for s in timestep_states]
        indicators = [[l.z for l in layers] for layers in cell_states]

        to_map = tf.concat((states, self.batch_out), axis=2)

        def map_output(elem):
            splits = tf.constant([elem_len, self._output_size])
            cell_states, outcome = array_ops.split(value=elem,
                                                   num_or_size_splits=splits,
                                                   axis=1)

            hs = [s.h for s in self.split_out_cell_states(cell_states)]
            gated = self.gate_input(tf.concat(hs, axis=1))
            embeded = self.embed_input(gated)
            loss, prediction = self.output_module(embeded, outcome)

            return tf.concat((loss, prediction), axis=1)

        mapped = tf.map_fn(map_output, to_map)

        loss = tf.reduce_mean(mapped[:, :, 0])
        predictions = mapped[:, :, 1:]
        train = tf.train.AdamOptimizer(1e-3).minimize(loss)

        return train, loss, indicators, predictions

    def train(self,
              batches_in,
              batches_out,
              load_vars_from_disk=False,
              epochs=3):

        truncate_len = len(batches_in[0])
        batch_size = len(batches_in[0][0])
        if self._graph.get(batch_size) is None:
            self._graph[batch_size] = self.network(batch_size, truncate_len)

        if self._session is None:
            self._session = tf.Session()

        if not load_vars_from_disk:
            init = tf.global_variables_initializer()
            self._session.run(init)
        else:
            self.load_variables()

        optim, loss, _, _ = self._graph[batch_size]
        for epoch in range(epochs):
            print('Epoch %d' % epoch)
            for batch_in, batch_out in zip(batches_in, batches_out):
                ops = [optim, loss]
                feed_dict = {
                    self.batch_in: batch_in,
                    self.batch_out: batch_out,
                }
                _, _loss = self._session.run(ops, feed_dict)
                print('loss:', _loss)

        self.save_variables()

    def predict(self, signal, variable_path=None):
        batch_size = signal.shape[1]
        truncate_len = signal.shape[0]

        if self._graph.get(batch_size) is None:
            self._graph[batch_size] = self.network(batch_size, truncate_len)

        if self._session is None:
            self.load_variables(variable_path)
            try:
                self.load_variables(variable_path)
            except:
                raise RuntimeError('Session unitialized and no variables saved'
                                   + ' at provided path %s' % variable_path)


        _, _, _, predictions = self._graph[batch_size]

        # batch_out is not used for prediction, but needs to be fed in
        batch_out_size = (signal.shape[0], signal.shape[1], self._output_size)

        _predictions = self._session.run(predictions, {
            self.batch_in: signal,
            self.batch_out: np.zeros(batch_out_size)
        })

        return np.array(_predictions)

    def predict_boundaries(self, signal):
        batch_size = signal.shape[1]
        truncate_len = signal.shape[0]

        if self._session is None:
            raise RuntimeError('Session unitialized; ' +
                               'try loading variables or training network')

        if self._graph.get(batch_size) is None:
            self._graph[batch_size] = self.network(batch_size, truncate_len)

        _, _, indicators, _ = self._graph[batch_size]

        # batch_out is not used for prediction, but needs to be fed in
        batch_out_size = (signal.shape[0], signal.shape[1], self._output_size)

        _indicators = self._session.run(indicators, {
            self.batch_in: signal,
            self.batch_out: np.zeros(batch_out_size)
        })

        return np.array(_indicators).T
