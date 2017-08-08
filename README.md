# hmlstm

This package implements the Hierarchical Multiscale LSTM network described by
Chung et al. in https://arxiv.org/abs/1609.01704

The network operates much like a normal multi-layerd RNN, with the addition of
boundary detection neturons. These are neurons in each layer that, ideally, 
learn to fire when there is a 'boundary' at the scale in the original signal
corresponding to that layer of the network.

### Installation

```
pip install git+https://github.com/n-s-f/hierarchical-rnn.git
```

Or, if you're interested in changing the code:

```
git clone https://github.com/n-s-f/hierarchical-rnn.git
cd hierarchical
python setup.py develop
```

### Character Classification

In this example, we'll consider the 
[text8](https://cs.fit.edu/~mmahoney/compression/textdata.html) data set, which
contains only lower case english characters, and spaces. We'll train on all
batches but the last, and test on just the last batch.

```.py
from hmlstm import HMLSTMNetwork, prepare_inputs, get_text

batches_in, batches_out = prepare_inputs(batch_size=10, truncate_len=5000, 
                                         step_size=2500, text_path='text8.txt')
                                         
network = HMLSTMNetwork(output_size=27, input_size=27, embed_size=2048, 
                        out_hidden_size=1024, hidden_state_sizes=1024, 
                        task='classification')

network.train(batches_in[:-1], batches_out[:-1], save_vars_to_disk=True, 
              load_vars_from_disk=False, variable_path='./text8')

predictions = network.predict(batches_in[-1], variable_path='./text8')
boundaries = network.predict_boundaries(batches_in[-1], variable_path='./text8')

# visualize boundaries
viz_char_boundaries(get_text(batches_out[-1][0]), get_text(predictions[0]), boundaries[0])
```

### Time series prediction

In this example, we'll do three-step-ahead prediction on a noisy set of signals
with sinusoidal activity at two scales.

```.py
from hmlstm import HMLSTMNetwork, convert_to_batches, plot_indicators

network = HMLSTMNetwork(input_size=1, task='regression', hidden_state_sizes=30,
                       embed_size=50, out_hidden_size=30, num_layers=2)
                       
# generate signals
num_signals = 300
signal_length = 400
x = np.linspace(0, 50 * np.pi, signal_length)
signals = [np.random.normal(0, .5, size=signal_length) +
           (2 * np.sin(.6 * x + np.random.random() * 10)) +
           (5 * np.sin(.1* x + np.random.random() * 10))
    for _ in range(num_signals)] 
    
batches_in, batches_out = convert_to_batches(signals, batch_size=10, steps_ahead=3)


network.train(batches_in[:-1], batches_out[:-1], save_vars_to_disk=True, 
              load_vars_from_disk=False, variable_path='./sinusoidal')

predictions = network.predict(batches_in[-1], variable_path='./sinusoidal')
boundaries = network.predict_boundaries(batches_in[-1], variable_path='./sinusoidal')

# visualize boundaries
plot_indicators(batches_out[-1][0], predictions[0], indicators=boundaries[0])
```

### Further information

Please see the doc strings in the code for more detailed documentation, and the
[demo notebook](https://github.com/n-s-f/hierarchical-rnn/blob/master/hmlstm_demo.ipynb)
for more thorough examples.

Pull requests or open github issues for improvements are very welcome.
