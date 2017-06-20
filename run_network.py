from hmlstm_network import HMLSTMNetwork
from text_input_utils import prepare_inputs


def get_network(batch_size, truncate_len):
    return HMLSTMNetwork(batch_size, 3, truncate_len, 29)


def train_network(batch_size, truncate_len):
    inputs = prepare_inputs(batch_size=batch_size, truncate_len=truncate_len)
    network = get_network(batch_size, truncate_len)
    network.train(*inputs)


if __name__ == '__main__':
    batch_size = 2
    truncate_len = 30

    inputs = prepare_inputs(batch_size=batch_size, truncate_len=truncate_len)
    network = get_network(batch_size, truncate_len)
    network.train(*inputs)

    prediction_network = get_network(batch_size=1, truncate_len=100000)
    inputs = prepare_inputs(batch_size=1, truncate_len=100000)
    prediction_network.predict_boundaries(inputs[0][0])
