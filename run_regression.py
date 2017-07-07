import numpy as np
import matplotlib.pyplot as plt
from hmlstm_network import HMLSTMNetwork
import tensorflow as tf
from text_input_utils import prepare_inputs
import tensorflow as tf
from string import ascii_lowercase

# simulate multiresolution data
num_signals = 20
signal_length = 4
x = np.linspace(0, 40 * np.pi, signal_length)
signals = [np.random.normal(0, 1, size=signal_length) 
           + (5 * np.sin(1 * x + np.random.random() * 100 * np.pi))
           + (5 * np.sin(.2 * x + np.random.random() * 100 * np.pi))
          for _ in range(num_signals)]

split = int(num_signals * .8)
train = signals[:split]
test = signals[split:]

# prepare data
train_batches_in = []
train_batches_out = []
batch_size = 2
start = 0
while start + batch_size < len(train):
    batch = train[start: start + batch_size]
    
    train_batches_in.append(np.array([s[:-1] for s in batch]).reshape(batch_size, -1, 1))
    train_batches_out.append(np.array([s[1:] for s in batch]).reshape(batch_size, -1, 1))

    start += batch_size
    
batch_size = 2
start = 0
test_batches_in = []
test_batches_out = []
while start + batch_size < len(test):
    batch = test[start: start + batch_size]
    
    test_batches_in.append(np.array([s[:-1] for s in batch]).reshape(batch_size, -1, 1))
    test_batches_out.append(np.array([s[1:] for s in batch]).reshape(batch_size, -1, 1))

    start += batch_size

network = HMLSTMNetwork(input_size=1, task='regression', hidden_state_sizes=[10, 20, 30],
                       embed_size=20, out_hidden_size=10, num_layers=3)

test_batches_in = np.swapaxes(np.array( test_batches_in ), 1, 2)
test_batches_out = np.swapaxes(np.array( test_batches_out ), 1, 2)
train_batches_in = np.swapaxes(np.array( train_batches_in ), 1, 2)
train_batches_out = np.swapaxes(np.array( train_batches_out ), 1, 2)

network.train(train_batches_in, train_batches_out, load_existing_vars=False, epochs=3)
writer = tf.summary.FileWriter('log/', graph=tf.get_default_graph())

boundaries = network.predict(test_batches_in[0][0])
boundaries = network.predict_boundaries(test_batches_in[0][0])
print(boundaries)
