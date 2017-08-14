import numpy as np
from fastdtw import fastdtw
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import coint

from cost_functions.huber_loss import huber_loss


class MyBaseline(object):
    def __init__(self, npz_path):
        super(MyBaseline, self).__init__()
        arr = np.load(npz_path)
        self.targets = arr['targets']
        self.inputs = arr['inputs']

        self.preds = np.empty(shape=self.targets.shape + (1,))
        # for ii, cur_in in enumerate(inputs):
        #    preds[ii] = cur_in[-1]  #broadcasting
        # but because we have the targets normed we simply do every prediction zero
        self.preds = np.zeros_like(self.preds)

        self.data_len = len(self.inputs)
        assert len(self.preds) == len(self.targets) and self.data_len == len(self.targets)

        self.mses = None
        self.huber_losses = None

    def getMSE(self):
        self.mses = np.empty(self.data_len)
        for ii, (pred, target) in enumerate(zip(self.preds, self.targets)):
            self.mses[ii] = mean_squared_error(pred, target)

        return np.mean(self.mses)

    def renderMSEs(self):
        f, ax = plt.subplots(figsize=(15, 7))
        ax.set(xscale="log")  # , yscale="log")
        sns.distplot(self.mses, ax=ax)
        # plt.show()

    def getHuberLoss(self):
        self.huber_losses = np.empty(self.data_len)
        for ii, (pred, target) in enumerate(zip(self.preds, self.targets)):
            self.huber_losses[ii] = np.mean(huber_loss(pred, target))

        return np.mean(self.huber_losses)

    def renderHuberLosses(self):
        f, ax = plt.subplots(figsize=(15, 7))
        ax.set(xscale="log")  # , yscale="log")
        sns.distplot(self.huber_losses, ax=ax)
        # plt.show()

    def get_dtw(self):
        dtw_scores = [fastdtw(self.targets[ind], self.preds[ind])[0] for ind in range(len(self.targets))]
        return np.mean(dtw_scores)

    # def get_cointegration(self):
    #     return coint(preds[0], targets[0])

    def renderRandomTargetVsPrediction(self):
        ind = np.random.randint(len(self.targets))
        fig = plt.figure(figsize=(15, 7))
        plt.plot(range(0, 60), self.inputs[ind].flatten(), 'r')
        plt.plot(range(60, 90), self.targets[ind].flatten(), 'b')
        plt.plot(range(60, 90), self.preds[ind].flatten(), 'g')
        plt.legend(['past', 'real future', 'predicted future'])
        # plt.show()
