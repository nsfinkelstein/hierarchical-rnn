from string import ascii_lowercase
from .hmlstm_network import HMLSTMNetwork
import re
import numpy as np


num_batches = 10
batch_size = 2
truncate_len = 10


def text(text_path, truncate_len, step_size):
    signals = load_text(text_path, truncate_len, step_size)

    print(signals)

    hot = [(one_hot_encode(intext), one_hot_encode(outtext))
           for intext, outtext in signals]

    return hot


def load_text(text_path, truncate_len, step_size):
    with open(text_path, 'r') as f:
        text = f.read()
        text = text.replace('\n', ' ')
        text = re.sub(' +', ' ', text).lower()

    # text = 'abcdefghijklmnopqrstuvwxyz' * 100000000

    signals = []
    start = 0
    for _ in range(batch_size * num_batches):
        intext = text[start:start + truncate_len]
        outtext = text[start + 1:start + truncate_len + 1]
        signals.append((intext, outtext))
        start += step_size

    return signals


def one_hot_encode(text):
    out = np.zeros((len(text), 29))

    def get_index(char):
        answers = {',': 26, '.': 27}

        if char in answers:
            return answers[char]

        try:
            return ascii_lowercase.index(char)
        except:
            return 28

    for i, t in enumerate(text):
        out[i, get_index(t)] = 1

    return out


def prepare_inputs(text_path='text.txt', batch_size=10, truncate_len=100,
                   step_size=None, num_batches=None):

    if step_size is None:
        step_size = truncate_len // 2

    y = text(text_path, truncate_len, step_size)

    if num_batches is None:
        num_batches = len(y) // batch_size

    batches_in = []
    batches_out = []

    for batch_number in range(num_batches):
        start = batch_number * batch_size
        end = start + batch_size
        batches_in.append([i for i, _ in y[start:end]])
        batches_out.append([o for _, o in y[start:end]])

    return batches_in, batches_out


def get_network():
    return HMLSTMNetwork(batch_size, 3, truncate_len, 29)


def run_everything():
    inputs = prepare_inputs()
    network = get_network()
    # print(network.predict(inputs[0][0]))
    # print(network.predict_boundaries(inputs[0][0]))
    network.train(*inputs)


if __name__ == '__main__':
    run_everything()
