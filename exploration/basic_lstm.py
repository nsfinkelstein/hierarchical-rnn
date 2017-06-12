import tensorflow as tf
import numpy as np
from numpy import random

# sizes
input_size = 10
state_size = 10


def lstm(x, cell_state, hidden_state):
    # set gate biases
    initial_bias = np.zeros((state_size, 1))
    forget_bias = tf.Variable(initial_bias, dtype=tf.float32)
    input_bias = tf.Variable(initial_bias, dtype=tf.float32)
    output_bias = tf.Variable(initial_bias, dtype=tf.float32)
    prop_bias = tf.Variable(initial_bias, dtype=tf.float32)

    # set gate weight
    initial_weight = random.rand(state_size, input_size + state_size)
    forget_weight = tf.Variable(initial_weight, dtype=tf.float32)
    input_weight = tf.Variable(initial_weight, dtype=tf.float32)
    output_weight = tf.Variable(initial_weight, dtype=tf.float32)
    prop_weight = tf.Variable(initial_weight, dtype=tf.float32)

    # process gates
    joint_input = tf.concat([x, hidden_state], axis=0)
    forget_gate = tf.sigmoid(
        tf.matmul(forget_weight, joint_input) + forget_bias)
    input_gate = tf.sigmoid(tf.matmul(input_weight, joint_input) + input_bias)
    output_gate = tf.sigmoid(
        tf.matmul(output_weight, joint_input) + output_bias)
    prop_gate = tf.tanh(tf.matmul(prop_weight, joint_input) + prop_bias)

    new_cell_state = tf.add(
        tf.multiply(forget_gate, cell_state),
        tf.multiply(input_gate, prop_gate))

    new_hidden_state = tf.multiply(output_gate, tf.tanh(new_cell_state))
    return new_cell_state, new_hidden_state


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
