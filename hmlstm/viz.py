import matplotlib.pyplot as plt


def plot_indicators(truth, prediction, indicators, steps_ahead=1):
    f, ax = plt.subplots()
    ax.plot(truth[steps_ahead:], label='truth')
    ax.plot(prediction, label='prediction')
    ax.legend()

    colors = ['r', 'b', 'g', 'o', 'm', 'l', 'c']
    for l, layer in enumerate(indicators):
        for i, indicator in enumerate(layer):
            if indicator == 1.:
                p = 1 / len(indicators)
                ymin = p * l
                ymax = p * (l + 1)
                ax.axvline(i, color=colors[l], ymin=ymin, ymax=ymax, alpha=.3)

    return f
