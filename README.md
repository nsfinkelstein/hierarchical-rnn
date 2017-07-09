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

### Usage

```.py
from hmlstm import HMLSTMNetwork

network = HMLSTMNetwork()

network.train(in_batches, out_batches)
network.predict(prediction_batch)
```

Please see the doc strings in the code for more detailed documentation, and the
[demo notebook](https://github.com/n-s-f/hierarchical-rnn/blob/master/hmlstm_demo.ipynb)
for more thorough examples.


