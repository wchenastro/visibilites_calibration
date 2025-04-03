import numpy as np
from matplotlib import pyplot as plt


def integrate(data, factor=1):
    integrated_data = data.reshape(
            np.hstack([data.shape[:-1], [-1, factor]])
            ).mean(axis = len(data.shape))

    return integrated_data


def plot_result(x1, y1, x2=None, y2=None, label1=None, label2=None, title=None):
    fig, ax1 = plt.subplots()

    if x2 is not None:
        ax1.plot(x1, y1, label=label1, color='blue')
        ax1.plot(x2, y2, label=label2, color='orange')
        y1_margin = np.max(np.abs(y1)) * 1.05
        ax1.set_ylim(-y1_margin, y1_margin)
    else:
        ax1.plot(x1, y1, label=label1, color='orange')
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Phase')
    ax1.set_title(title)
    if x2 is not None:
        ax1.legend()

    fig.tight_layout()
