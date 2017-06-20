from .hmlstm_cell import HMLSTMCell
from .multi_hmlstm_cell import MultiHMLSTMCell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf


class MultiHMLSTMNetwork(object):
    def __init__(self,
                 batch_size,
                 num_layers,
                 truncate_len,
                 num_units,
                 save_path='./hmlstm.ckpt'):
        self._out_hidden_size = 100
        self._embed_size = 100
        self._batch_size = batch_size
        self._num_layers = num_layers
        self._truncate_len = truncate_len
        self._num_units = num_units  # the length of c and h
        self._save_path = save_path

        batch_shape = (batch_size, truncate_len, num_units)
        self.batch_in = tf.placeholder(
            tf.float32, shape=batch_shape, name='batch_in')
        self.batch_out = tf.placeholder(
            tf.int32, shape=batch_shape, name='batch_out')

        self._initialize_output_variables()
        self._initialize_gate_variables()

        self.optim, self.loss, self.indicators, self.predictions = self.create_network(
            self.create_output_module)

    def _initialize_gate_variables(self):
        with vs.variable_scope('gates'):
            for l in range(self._num_layers):
                vs.get_variable(
                    'gate_%s' % l, [self._num_units * self._num_layers, 1],
                    dtype=tf.float32)

    def _initialize_output_variables(self):
        with vs.variable_scope('output_module'):
            vs.get_variable('b1', [1, self._out_hidden_size], dtype=tf.float32)
            vs.get_variable('b2', [1, self._out_hidden_size], dtype=tf.float32)
            vs.get_variable('b3', [1, self._num_units], dtype=tf.float32)
            vs.get_variable(
                'w1', [self._embed_size, self._out_hidden_size],
                dtype=tf.float32)
            vs.get_variable(
                'w2', [self._out_hidden_size, self._out_hidden_size],
                dtype=tf.float32)
            vs.get_variable(
                'w3', [self._out_hidden_size, self._num_units],
                dtype=tf.float32)
            embed_shape = [
                self._num_layers * self._num_units, self._embed_size
            ]
            vs.get_variable('embed_weights', embed_shape, dtype=tf.float32)

    def gate_input(self, hidden_states):
        # gate the incoming hidden states
        with vs.variable_scope('gates', reuse=True):
            gates = []
            for l in range(self._num_layers):
                weights = vs.get_variable(
                    'gate_%s' % l, [self._num_units * self._num_layers, 1],
                    dtype=tf.float32)
                gates.append(tf.sigmoid(tf.matmul(hidden_states, weights)))

            split = array_ops.split(
                value=hidden_states,
                num_or_size_splits=self._num_layers,
                axis=1)
            gated_list = []
            for gate, hidden_state in zip(gates, split):
                gated_list.append(tf.multiply(gate, hidden_state))

            gated_input = tf.concat(gated_list, axis=1)
        return gated_input

    def create_output_module(self, gated_input, time_step):
        with vs.variable_scope('output_module', reuse=True):

            in_size = self._num_layers * self._num_units
            embed_shape = [in_size, self._embed_size]
            embed_weights = vs.get_variable(
                'embed_weights', embed_shape, dtype=tf.float32)

            b1 = vs.get_variable(
                'b1', [1, self._out_hidden_size], dtype=tf.float32)
            b2 = vs.get_variable(
                'b2', [1, self._out_hidden_size], dtype=tf.float32)
            b3 = vs.get_variable('b3', [1, self._num_units], dtype=tf.float32)
            w1 = vs.get_variable(
                'w1', [self._embed_size, self._out_hidden_size],
                dtype=tf.float32)
            w2 = vs.get_variable(
                'w2', [self._out_hidden_size, self._out_hidden_size],
                dtype=tf.float32)
            w3 = vs.get_variable(
                'w3', [self._out_hidden_size, self._num_units],
                dtype=tf.float32)

            # embedding
            prod = tf.matmul(gated_input, embed_weights)
            embedding = tf.nn.relu(prod)

            # feed forward network
            # first layer
            l1 = tf.nn.tanh(tf.matmul(embedding, w1) + b1)

            # second layer
            l2 = tf.nn.tanh(tf.matmul(l1, w2) + b2)

            # the loss function used below
            # softmax_cross_entropy_with_logits
            prediction = tf.matmul(l2, w3) + b3

            loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.batch_out[:, time_step:time_step + 1, :],
                logits=prediction,
                name='loss')
            scalar_loss = tf.reduce_mean(loss, name='loss_mean')
        return scalar_loss, prediction

    def create_network(self, output_module):
        def hmlstm_cell():
            return HMLSTMCell(self._num_units, self._batch_size)

        hmlstm = MultiHMLSTMCell(
            [hmlstm_cell() for _ in range(self._num_layers)])

        state = hmlstm.zero_state(self._batch_size, tf.float32)
        ha_shape = [self._batch_size, 1, (self._num_layers * self._num_units)]
        h_aboves = tf.zeros(ha_shape)

        loss = tf.constant(0.0)
        indicators = []
        predictions = []
        for i in range(self._truncate_len):
            inputs = array_ops.concat(
                (self.batch_in[:, i:(i + 1), :], h_aboves), axis=2)

            hidden_states, state = hmlstm(inputs, state)
            concated_hs = array_ops.concat(hidden_states[1:], axis=1)
            h_aboves = array_ops.concat(
                [
                    concated_hs, tf.zeros(
                        [self._batch_size, self._num_units], dtype=tf.float32)
                ],
                axis=1)
            h_aboves = tf.expand_dims(h_aboves, 1)

            gated = self.gate_input(array_ops.concat(hidden_states, axis=1))
            new_loss, new_prediction = output_module(gated, i)
            loss += tf.reduce_mean(new_loss)

            predictions.append(new_prediction)
            indicators += [z for c, h, z in state]

        train = tf.train.AdamOptimizer(1e-3).minimize(loss)
        return train, loss, indicators, predictions

    def train(self, batches_in, batches_out, fresh_start=True, epochs=10):
        saver = tf.train.Saver()

        with tf.Session() as sess:

            if fresh_start:
                init = tf.global_variables_initializer()
                sess.run(init)
            elif not fresh_start:
                print('loading variables...')
                saver.restore(sess, self._save_path)

            for epoch in range(epochs):
                print('Epoch %d' % epoch)
                for batch_in, batch_out in zip(batches_in, batches_out):
                    ops = [self.optim, self.loss]
                    feed_dict = {
                        self.batch_in: batch_in,
                        self.batch_out: batch_out,
                    }
                    _, _loss = sess.run(ops, feed_dict)
                    print('loss:', _loss)

            print('saving variables...')
            saver.save(sess, self._save_path)

    def predict(self, batch_in):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print('loading variables...')
            saver.restore(sess, self._save_path)

            return sess.run(self.predictions, {
                self.batch_in: batch_in,
            })

    def predict_boundaries(self, batch_in):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print('loading variables...')
            saver.restore(sess, self._save_path)

            return sess.run(self.indicators, {
                self.batch_in: batch_in,
            })
