# hmlstm

This package implements the Hierarchical Multiscale LSTM network described by
Chung et. al in https://arxiv.org/abs/1609.01704

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


