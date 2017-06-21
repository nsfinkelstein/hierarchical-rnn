from hmlstm_network import HMLSTMNetwork
from text_input_utils import prepare_inputs

batch_size = 2
truncate_len = 3
num_batches = 3

network = HMLSTMNetwork(num_layers=3, output_size=29,
                        hidden_state_sizes=[50, 40, 30])

inputs = prepare_inputs(
    batch_size=batch_size, truncate_len=truncate_len, num_batches=3)

print(len(inputs))
network.train(*inputs)

inputs = prepare_inputs(batch_size=1, truncate_len=100)
print(network.predict_boundaries(inputs[0][2], reuse=True))
