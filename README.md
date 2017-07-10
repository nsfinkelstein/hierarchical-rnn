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

### Character Classification

In this example, we'll consider the 
[text8](https://cs.fit.edu/~mmahoney/compression/textdata.html) data set, which
contains only lower case english characters, and spaces. We'll train on all
batches but the last, and test on just the last batch.

```.py
from hmlstm import HMLSTMNetwork, prepare_inputs, get_text

batches_in, batches_out = prepare_inputs(batch_size=10, truncate_len=5000, 
                                         step_size=2500, text_path='text8.txt')
                                         
network = HMLSTMNetwork(input_size=)

network.train(in_batches[:-1], out_batches[:-1])

predictions = network.predict(in_batches[-1])
boundaries = network.predict_boundaries(in_batches[-1])
```

### Time series prediction

___

Please see the doc strings in the code for more detailed documentation, and the
[demo notebook](https://github.com/n-s-f/hierarchical-rnn/blob/master/hmlstm_demo.ipynb)
for more thorough examples.


