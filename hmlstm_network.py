from hmlstm_cell import HMLSTMCell, HMLSTMState
from multi_hmlstm_cell import MultiHMLSTMCell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python import debug as tf_debug
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
                 task='classification',
                 network='dynamic',
                 save_path='./hmlstm.ckpt'):

        self._out_hidden_size = out_hidden_size
        self._embed_size = embed_size
        self._num_layers = num_layers
        self._save_path = save_path
        self._input_size = input_size
        self._training_graph = dict()
        self._prediction_graph = None
        self._network = network

        if type(hidden_state_sizes) == int:
            self._hidden_state_sizes = [hidden_state_sizes] * self._num_layers
        else:
            self._hidden_state_sizes = hidden_state_sizes

        if task == 'classification':
            self._loss_function = tf.nn.softmax_cross_entropy_with_logits
            self._output_size = output_size
            self._prediction_arg = 'logits'
        elif task == 'regression':
            self._loss_function = tf.losses.mean_squared_error
            self._output_size = 1
            self._prediction_arg = 'predictions'
        else:
            raise ValueError('Not a valid task')

        batch_shape = (None, None, self._output_size)
        self.batch_in = tf.placeholder(
            tf.float32, shape=batch_shape, name='batch_in')
        self.batch_out = tf.placeholder(
            tf.int32, shape=batch_shape, name='batch_out')

        self._optimizer = tf.train.AdamOptimizer(1e-3)
        self._initialize_output_variables()
        self._initialize_gate_variables()
        self._initialize_embedding_variables()

    def _initialize_gate_variables(self):
        with vs.variable_scope('gates_vars'):
            for l in range(self._num_layers):
                vs.get_variable(
                    'gate_%s' % l, [sum(self._hidden_state_sizes), 1],
                    dtype=tf.float32)

    def _initialize_embedding_variables(self):
        with vs.variable_scope('embedding_vars'):
            embed_shape = [sum(self._hidden_state_sizes), self._embed_size]
            vs.get_variable('embed_weights', embed_shape, dtype=tf.float32)

    def _initialize_output_variables(self):
        with vs.variable_scope('output_module_vars'):
            vs.get_variable('b1', [1, self._out_hidden_size], dtype=tf.float32)
            vs.get_variable('b2', [1, self._out_hidden_size], dtype=tf.float32)
            vs.get_variable('b3', [1, self._output_size], dtype=tf.float32)
            vs.get_variable(
                'w1', [self._embed_size, self._out_hidden_size],
                dtype=tf.float32)
            vs.get_variable(
                'w2', [self._out_hidden_size, self._out_hidden_size],
                dtype=tf.float32)
            vs.get_variable(
                'w3', [self._out_hidden_size, self._output_size],
                dtype=tf.float32)

    def gate_input(self, hidden_states):
        # gate the incoming hidden states
        with vs.variable_scope('gates_vars', reuse=True):
            gates = []
            for l in range(self._num_layers):
                weights = vs.get_variable('gate_%d' % l, dtype=tf.float32)
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
        with vs.variable_scope('embedding_vars', reuse=True):
            embed_weights = vs.get_variable('embed_weights', dtype=tf.float32)

            prod = tf.matmul(gated_input, embed_weights)
            embedding = tf.nn.relu(prod)

        return embedding

    def output_module(self, embedding, time_step):
        with vs.variable_scope('output_module_vars', reuse=True):
            b1 = vs.get_variable('b1')
            b2 = vs.get_variable('b2')
            b3 = vs.get_variable('b3')
            w1 = vs.get_variable('w1')
            w2 = vs.get_variable('w2')
            w3 = vs.get_variable('w3')

            # feed forward network
            # first layer
            l1 = tf.nn.tanh(tf.matmul(embedding, w1) + b1)

            # second layer
            l2 = tf.nn.tanh(tf.matmul(l1, w2) + b2)

            # the loss function used below
            # softmax_cross_entropy_with_logits
            prediction = tf.add(tf.matmul(l2, w3), b3, name='prediction')

            loss_args = {
                self._prediction_arg: tf.squeeze(prediction),
                'labels': tf.squeeze(self.batch_out[:, time_step:time_step + 1, :]),
            }

            loss = self._loss_function(**loss_args)
            scalar_loss = tf.reduce_mean(loss, name='loss_mean')
        return scalar_loss, prediction

    def create_multicell(self, batch_size, reuse):
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
                              h_below_size, h_above_size, reuse)

        hmlstm = MultiHMLSTMCell(
            [hmlstm_cell(l) for l in range(self._num_layers)], reuse=reuse)

        return hmlstm

    def dynamic_network(self, output_module, batch_size, truncate_len, reuse):
        hmlstm = self.create_multicell(batch_size, reuse)

        def scan_func(accum, elem):
            # each element is the set of all hidden states from the previous
            # time step
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

            h_aboves = self.get_h_aboves([cs.h for cs in cell_states],
                                         batch_size, hmlstm)

            hmlstm_in = array_ops.concat((elem, h_aboves), axis=2)
            _, state = hmlstm(hmlstm_in, cell_states)

            return array_ops.concat([array_ops.concat(s, axis=1)
                                     for s in state], axis=1)

        elem_len = (sum(self._hidden_state_sizes) * 2) + self._num_layers
        initial = tf.zeros([batch_size, elem_len])

        elems = tf.unstack(self.batch_in, num=truncate_len, axis=1)
        squeezed_elems = [tf.squeeze(e) for e in elems]

        hidden_states = tf.scan(scan_func, squeezed_elems, initial)
        print(hidden_states)

    def unrolled_network(self, output_module, batch_size, truncate_len, reuse):
        hmlstm = self.create_multicell(batch_size, reuse)

        state = hmlstm.zero_state(batch_size, tf.float32)
        ha_shape = [batch_size, 1, sum(self._hidden_state_sizes)]
        h_aboves = tf.zeros(ha_shape, name='h_aboves')

        loss = tf.constant(0.0, name='loss')
        indicators = []
        predictions = []
        for i in range(truncate_len):
            inputs = array_ops.concat(
                (self.batch_in[:, i:(i + 1), :], h_aboves), axis=2,
                name='concat_value_and_h_aboves')

            hidden_states, state = hmlstm(inputs, state)

            h_aboves = self.get_h_aboves(hidden_states, batch_size, hmlstm)

            gated = self.gate_input(array_ops.concat(hidden_states, axis=1))
            embeded = self.embed_input(gated)
            new_loss, new_prediction = output_module(embeded, i)
            loss += new_loss

            predictions.append(new_prediction)
            indicators.append([s.z for s in state])

        train = self._optimizer.minimize(loss)
        return train, loss, indicators, predictions

    def network(self, *args, **kwargs):
        if self._network == 'dynamic':
            return self.dynamic_network(*args, **kwargs)
        elif self._network == 'unrolled':
            return self.unrolled_network(*args, **kwargs)

    def get_h_aboves(self, hidden_states, batch_size, hmlstm):
        concated_hs = array_ops.concat(hidden_states[1:], axis=1, name='concat_h_aboves')

        h_above_for_last_layer = tf.zeros(
            [batch_size, hmlstm._cells[-1]._h_above_size], dtype=tf.float32
            , name='habove_for_last_layer')

        h_aboves = array_ops.concat(
            [concated_hs, h_above_for_last_layer], axis=1, name='final_h_aboves')

        h_aboves = tf.expand_dims(h_aboves, 1)

        return h_aboves


    def train(self,
              batches_in,
              batches_out,
              reuse=None,
              load_existing_vars=False,
              epochs=3):

        batch_size = len(batches_in[0])
        truncate_len = len(batches_in[0][0])
        key = (batch_size, truncate_len)
        if self._training_graph.get(key) is None:
            self._training_graph[key] = self.network(self.output_module,
                                                       batch_size,
                                                       truncate_len, reuse)

        optim, loss, _, _ = self._training_graph[key]

        saver = tf.train.Saver()
        with tf.Session() as sess:

            if not load_existing_vars:
                init = tf.global_variables_initializer()
                sess.run(init)
            elif load_existing_vars:
                print('loading variables...')
                saver.restore(sess, self._save_path)

            for epoch in range(epochs):
                print('Epoch %d' % epoch)
                for batch_in, batch_out in zip(batches_in, batches_out):
                    ops = [optim, loss]
                    feed_dict = {
                        self.batch_in: batch_in,
                        self.batch_out: batch_out,
                    }
                    _, _loss = sess.run(ops, feed_dict)
                    print('loss:', _loss)

            print('saving variables...')
            saver.save(sess, self._save_path)

    def predict(self, signal, reuse=True):
        batch_size = 1
        truncate_len = len(signal)
        key = (batch_size, truncate_len)
        if self._training_graph.get(key) is None:
            self._training_graph[key] = self.network(self.output_module,
                                                       batch_size,
                                                       truncate_len, reuse)


        _, _, _, predictions = self._training_graph[key]

        saver = tf.train.Saver()
        with tf.Session() as sess:
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # sess.add_tensor_filter('debug', lambda d, t: 'xxx' in d.node_name)

            print('loading variables...')
            saver.restore(sess, self._save_path)

            _predictions = sess.run(predictions, {
                self.batch_in: [signal],
            })

        return np.array(_predictions)

    def predict_boundaries(self, signal, reuse=True):
        batch_size = 1
        truncate_len = len(signal)
        key = (batch_size, truncate_len)
        if self._training_graph.get(key) is None:
            self._training_graph[key] = self.network(self.output_module,
                                                       batch_size,
                                                       truncate_len, reuse)

        _, _, indicators, _ = self._training_graph[key]

        saver = tf.train.Saver()
        with tf.Session() as sess:
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # sess.add_tensor_filter('debug', lambda d, t: 'xxx' in d.node_name)

            print('loading variables...')
            saver.restore(sess, self._save_path)

            _indicators = sess.run(indicators, {
                self.batch_in: [signal],
            })

        return np.array(_indicators).T
