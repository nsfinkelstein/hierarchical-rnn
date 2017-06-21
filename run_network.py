from hmlstm_network import HMLSTMNetwork
from text_input_utils import prepare_inputs

batch_size = 2
truncate_len = 100
num_layers = 3
num_batches = 20

network = HMLSTMNetwork(num_layers=num_layers, output_size=29, input_size=29,
                        embed_size=1024, out_hidden_size=500,
                        hidden_state_sizes=[300, 400, 500])

inputs = prepare_inputs(
    batch_size=batch_size, truncate_len=truncate_len, num_batches=num_batches)

print(len(inputs[0]))
network.train(*inputs, epochs=1)

inputs = prepare_inputs(batch_size=1, truncate_len=10000)
print(network.predict_boundaries(inputs[0][1], reuse=True))
