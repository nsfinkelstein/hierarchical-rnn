import tensorflow as tf
import numpy as np


# sizes
def get_operation(z, z_below):
    # TODO: Make sure this is actually checking for equality
    if z == 1:
        return 'flush'

    if z == 0 and z_below == 1:
        return 'update'

    if z == 0 and z_below == 0:
        return 'copy'

    raise ValueError('Invalid indicators')


def hmlstm_layer(c, h, h_below, h_above, z, z_below, state_size=200):
    # figure out operation
    operation = get_operation(z, z_below)

    if operation == 'copy':
        return c, h, z

    # set biases and weights
    weight_size = (state_size * 4) + 1
    b = tf.Variable(np.zeros((weight_size, 1)), dtype=tf.float32)
    w = tf.Variable(np.random.rand(weight_size, state_size), dtype=tf.float32)
    wb = tf.Variable(np.random.rand(weight_size, state_size), dtype=tf.float32)
    wa = tf.Variable(np.random.rand(weight_size, state_size), dtype=tf.float32)

    # process gates
    joint_input = tf.add(
        tf.add(tf.matmul(w, h), tf.multiply(z, tf.matmul(wa, h_above))),
        tf.add(tf.multiply(z_below, tf.matmul(wb, h_below)), b)
    )

    f = tf.sigmoid(joint_input[:state_size])
    i = tf.sigmoid(joint_input[state_size: 2 * state_size])
    o = tf.sigmoid(joint_input[2 * state_size: 3 * state_size])
    g = tf.tanh(joint_input[3 * state_size: 4 * state_size])
    z_tilde = tf.contrib.keras.backend.hard_sigmoid(joint_input[-1])

    new_z = tf.constant(np.random.binomial(1, z_tilde))

    if operation == 'update':
        new_c = tf.add(tf.multiply(f, c), tf.multiply(i, g))

    if operation == 'flush':
        new_c = tf.multiply(i, g)

    new_h = tf.multiply(o, tf.tanh(new_c))
    return new_c, new_h, new_z


# set placeholders
x = tf.placeholder(tf.float32, shape=(input_size, 1))
hidden_state = tf.placeholder(tf.float32, shape=(state_size, 1))
cell_state = tf.placeholder(tf.float32, shape=(state_size, 1))
cs, hs = lstm(x, cell_state, hidden_state)

constants = {
    x: random.rand(input_size, 1),
    cell_state: np.zeros((state_size, 1)),
    hidden_state: np.zeros((state_size, 1)),
}
session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
for _ in range(200):
    cos, hos = session.run([cs, hs], constants)
    constants = {
        x: random.rand(input_size, 1),
        cell_state: cos,
        hidden_state: hos,
    }
    print(cos, hos)
