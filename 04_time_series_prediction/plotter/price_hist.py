import numpy as np
from matplotlib import pyplot as plt


def renderRandomMultipleTargetsVsPredictions(targets, inputs, preds):
    INDEX_PRICE_INPUT = 0
    # fig = plt.figure(figsize=(15, 7))
    fig, axes = plt.subplots(3, 3, figsize=(15, 7))
    # print type(axes)
    # print axes.shape
    for ax in axes.flatten():
        ind = np.random.randint(len(targets))
        ax.plot(range(0, 60), inputs[ind, :, INDEX_PRICE_INPUT].flatten(), 'r', label='past')
        ax.plot(range(60, 90), targets[ind].flatten(), 'b', label='real future')
        ax.plot(range(60, 90), preds[ind].flatten(), 'g', label='predicted future')
        plt.legend(loc='best')
