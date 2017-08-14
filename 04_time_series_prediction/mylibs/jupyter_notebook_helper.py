# -*- coding: utf-8 -*-
"""Jupyter Notebook Helpers"""

from IPython.display import display, HTML
import datetime
# from time import time
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from collections import OrderedDict
import operator


def show_graph(graph_def, frame_size=(900, 600)):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:{height}px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(height=frame_size[1], data=repr(str(graph_def)), id='graph' + timestamp)
    iframe = """
        <iframe seamless style="width:{width}px;height:{height}px;border:0" srcdoc="{src}"></iframe>
    """.format(width=frame_size[0], height=frame_size[1] + 20, src=code.replace('"', '&quot;'))
    display(HTML(iframe))


def getRunTime(function):  # a = lambda _ = None : 3 or #a = lambda : 3
    run_start_time = time.time()
    result = function()
    run_time = time.time() - run_start_time
    return result, run_time


def getWriter(key, graph, folder):
    # tensorboard --logdir=<folder>

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return tf.summary.FileWriter(
        logdir=os.path.join(folder, timestamp, key),
        graph=graph
    )


def plotStats(stats, keys, stats_interval=1):
    # Plot the change in the validation and training set error over training.

    # stats[0:, keys[k]] #0 epoch number
    # stats[1:, keys[k]] #1 for training and validation
    # keys is from string to index
    # stats shape is [epochs, 4]

    fig_1 = plt.figure(figsize=(12, 6))
    ax_1 = fig_1.add_subplot(111)

    # ax_1.hold(True)
    for k in ['error(train)', 'error(valid)']:
        ax_1.plot(np.arange(1, stats.shape[0]) * stats_interval,
                  stats[1:, keys[k]], label=k)
    # ax_1.hold(False)

    ax_1.legend(loc=0)
    ax_1.set_xlabel('Epoch number')

    # Plot the change in the validation and training set accuracy over training.
    fig_2 = plt.figure(figsize=(12, 6))
    ax_2 = fig_2.add_subplot(111)

    # ax_2.hold(True)
    for k in ['acc(train)', 'acc(valid)']:
        ax_2.plot(np.arange(1, stats.shape[0]) * stats_interval,
                  stats[1:, keys[k]], label=k)
    # ax_2.hold(False)

    ax_2.legend(loc=0)
    ax_2.set_xlabel('Epoch number')
    # plt.show() better do it outside when you want it
    return fig_1, ax_1, fig_2, ax_2


def initStats(epochs):
    stats = np.zeros((epochs, 4))

    keys = {
        'error(train)': 0,
        'acc(train)': 1,
        'error(valid)': 2,
        'acc(valid)': 3
    }

    return stats, keys


def gatherStats(e, train_error, train_accuracy, valid_error, valid_accuracy, stats):
    stats[e, 0] = train_error
    stats[e, 1] = train_accuracy
    stats[e, 2] = valid_error
    stats[e, 3] = valid_accuracy

    return stats


class DynStats(object):
    """
    dynStats = DynStats()
    dynStats.gatherStats(train_error, train_accuracy, valid_error, valid_accuracy)
    return dynStats.stats, dynStats.keys
    """

    def __init__(self, accuracy=False, validation=False):
        super(DynStats, self).__init__()
        self.__stats = []

        keys = self.KEYS.copy()

        if accuracy is False:
            acc_keys = self.__getAccuracyKeys(keys=keys)
            for acc_key in acc_keys:
                del keys[acc_key]

        if validation is False:
            valid_keys = self.__getValidationKeys(keys=keys)
            for valid_key in valid_keys:
                del keys[valid_key]

        # renumber them
        counter = 0
        for cur_key in self.KEYS.keys():
            if cur_key in keys.keys():
                keys[cur_key] = counter
                counter += 1

        self.keys = keys

    KEYS = OrderedDict([
        ('error(train)', 0),
        ('acc(train)', 1),
        ('error(valid)', 2),
        ('acc(valid)', 3),
    ])

    def gatherStats(self, train_error, train_accuracy=None, valid_error=None, valid_accuracy=None):
        """# KEEP THE ORDER"""
        cur_stats = [train_error]
        if train_accuracy is not None:
            cur_stats.append(train_accuracy)
        if valid_error is not None:
            cur_stats.append(valid_error)
        if valid_accuracy is not None:
            cur_stats.append(valid_accuracy)

        self.__stats.append(np.array(cur_stats))

        return self.stats

    @property
    def stats(self):
        return np.array(self.__stats)

    @staticmethod
    def __getAccuracyKeys(keys):
        return [key for key in keys.keys() if "acc" in key]

    @staticmethod
    def __getValidationKeys(keys):
        return [key for key in keys.keys() if "valid" in key]

    @staticmethod
    def __getErrorKeys(keys):
        return [key for key in keys.keys() if "error" in key]

    def plotStats(self, stats_interval=1):
        # Plot the change in the validation and training set error over training.

        # stats[0:, keys[k]] #0 epoch number
        # stats[1:, keys[k]] #1 for training and validation
        # keys is from string to index
        # stats shape is [epochs, 4]

        fig_1 = plt.figure(figsize=(12, 6))
        ax_1 = fig_1.add_subplot(111)

        # ax_1.hold(True)
        for kk in self.__getErrorKeys(keys=self.keys):
            ax_1.plot(np.arange(1, self.stats.shape[0]) * stats_interval, self.stats[1:, self.keys[kk]], label=kk)
        # ax_1.hold(False)

        ax_1.legend(loc=0)
        ax_1.set_xlabel('Epoch number')

        figs = [fig_1]
        axes = [ax_1]

        # Plot the change in the validation and training set accuracy over training.
        accuracyKeys = self.__getAccuracyKeys(keys=self.keys)
        if len(accuracyKeys) > 0:
            fig_2 = plt.figure(figsize=(12, 6))
            ax_2 = fig_2.add_subplot(111)

            # ax_2.hold(True)
            for kk in accuracyKeys:
                ax_2.plot(np.arange(1, self.stats.shape[0]) * stats_interval, self.stats[1:, self.keys[kk]], label=kk)
            # ax_2.hold(False)

            ax_2.legend(loc=0)
            ax_2.set_xlabel('Epoch number')

            figs.append(fig_2)
            axes.append(ax_2)

        # plt.show() better do it outside when you want it

        return figs, axes


def renderStatsList(stats_list, epochs, title='Training Error', kk='error(train)'):
    fig = plt.figure(figsize=(12, 6))

    assert len(stats_list) > 0, "the stats list should not be empty, a nothing experiment does not make sense"
    keys = stats_list[0].keys
    valid_err_ind = keys['error(valid)']

    min_valid_errs = [np.min(stat.stats[:, valid_err_ind]) for stat in stats_list]

    best_stats_locations = np.argsort(min_valid_errs)[
                           :7]  # only seven because these are the colors support by default by matplotlib

    for ii, cur_stat in enumerate(stats_list):
        stats = cur_stat.stats
        xValues = np.arange(1, stats.shape[0])
        yValues = stats[1:, keys[kk]]

        if ii in best_stats_locations:
            plt.plot(xValues, yValues)
        else:
            plt.plot(xValues, yValues, c='lightgrey')

    plt.legend(loc=0)
    plt.title(title + ' over {} epochs'.format(epochs))
    plt.xlabel('Epoch number')
    plt.ylabel(title)
    plt.grid()

    return fig  # fig.savefig('cw%d_part%d_%02d_fig.svg' % (coursework, part, figcount))


def renderStatsCollection(statsCollection, label_texts, title='Training Error', kk='error(train)', keys=DynStats.KEYS):
    """
    usage:
    statsCollection[(state_size, num_steps)] = stats
    (note that the stats above are NOT dynamic stats)

    label_texts depends on how many keys you have in your collection
    """

    fig = plt.figure(figsize=(12, 6))

    epochs = len(statsCollection.values()[0])

    maxValidAccs = OrderedDict([(key, max(val[:, -1])) for key, val in statsCollection.iteritems()])
    highValidAccs = sorted(maxValidAccs.items(), key=operator.itemgetter(1))[::-1]
    highValidAccs = OrderedDict(
        highValidAccs[:7])  # only seven because these are the colors support by default by matplotlib

    for key in statsCollection:
        label = ", ".join(
            [(label_texts[ii] + ": " + str(val)) for ii, val in enumerate(key)]
        )
        stats = statsCollection[key]
        xValues = np.arange(1, stats.shape[0])
        yValues = stats[1:, keys[kk]]

        if key in highValidAccs.keys():
            plt.plot(xValues, yValues, label=label)
        else:
            plt.plot(xValues, yValues, c='lightgrey')
            # plt.hold(True)

    # plt.hold(False)
    plt.legend(loc=0)
    plt.title(title + ' over {} epochs'.format(epochs))
    plt.xlabel('Epoch number')
    plt.ylabel(title)
    plt.grid()
    plt.show()

    return fig  # fig.savefig('cw%d_part%d_%02d_fig.svg' % (coursework, part, figcount))


def renderStatsListWithLabels(stats_list, label_text, title='Training Error', kk='error(train)'):
    keys = [(s,) for s in range(len(stats_list))]
    stats_dict = OrderedDict(zip(keys, [cur_stat.stats for cur_stat in stats_list]))

    return renderStatsCollection(kk=kk, title=title, label_texts=[label_text], statsCollection=stats_dict,
                                 keys=stats_list[0].keys)


def renderStatsCollectionOfCrossValids(stats_dic, label_texts, title='Training Error', kk='error(train)',
                                       drop_firsts=1, with_individual_folds=True):
    """
    usage:
    stats_dic[(learning_rate,)] = stats_list
    (note that the stats above are indeed dynamic stats)

    label_texts depends on how many keys you have in your collection
    """
    fig = plt.figure(figsize=(12, 6))

    num_k_folds = len(stats_dic.values()[0])
    epochs = len(stats_dic.values()[0][0].stats)
    keys = stats_dic.values()[0][0].keys

    # maxValidAccs = OrderedDict([(key, max(val[:, -1])) for key, val in statsCollection.iteritems()])
    # highValidAccs = sorted(maxValidAccs.items(), key=operator.itemgetter(1))[::-1]
    # highValidAccs = OrderedDict(
    #     highValidAccs[:7])  # only seven because these are the colors support by default by matplotlib

    kinds = stats_dic.keys()  # ('cubic', 'quadratic', 'slinear', 'nearest', 'linear', 'zero', 4, 5)
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, len(kinds)))
    # https://stackoverflow.com/questions/7513262/matplotlib-large-set-of-colors-for-plots

    for color, hyper_params_key in zip(colors, stats_dic.keys()):
        label = ", ".join(
            [(label_texts[ii] + ": " + str(val)) for ii, val in enumerate(hyper_params_key)]
        )

        x_values = np.arange(drop_firsts, epochs)

        dyn_stats_list = stats_dic[hyper_params_key]

        if with_individual_folds:
            for jj, dyn_stats in enumerate(dyn_stats_list):
                stats = dyn_stats.stats
                y_values = stats[drop_firsts:, keys[kk]]
                # if jj == 0:
                #     plt.plot(x_values, y_values, label=label, c=color)
                # else:
                plt.plot(x_values, y_values, c='lightgrey')

        y_values_list = np.array([dyn_stats.stats[drop_firsts:, keys[kk]] for dyn_stats in dyn_stats_list])
        y_values_mean = np.mean(y_values_list, axis=0)
        plt.plot(x_values, y_values_mean, label=label, c=color)

    # plt.hold(False) #deprecated
    plt.legend(loc=0)
    plt.title(title + ' over {} epochs'.format(epochs))
    plt.xlabel('Epoch number')
    plt.ylabel(title)
    plt.grid()

    return fig  # fig.savefig('cw%d_part%d_%02d_fig.svg' % (coursework, part, figcount))


def my_plot_convergence(*args, **kwargs):
    """Plot one or several convergence traces.

    Parameters
    ----------
    * `args[i]` [`OptimizeResult`, list of `OptimizeResult`, or tuple]:
        The result(s) for which to plot the convergence trace.

        - if `OptimizeResult`, then draw the corresponding single trace;
        - if list of `OptimizeResult`, then draw the corresponding convergence
          traces in transparency, along with the average convergence trace;
        - if tuple, then `args[i][0]` should be a string label and `args[i][1]`
          an `OptimizeResult` or a list of `OptimizeResult`.

    * `ax` [`Axes`, optional]:
        The matplotlib axes on which to draw the plot, or `None` to create
        a new one.

    * `true_minimum` [float, optional]:
        The true minimum value of the function, if known.

    * `yscale` [None or string, optional]:
        The scale for the y-axis.

    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    # <3 legacy python
    ax = kwargs.get("ax", None)
    true_minimum = kwargs.get("true_minimum", None)
    yscale = kwargs.get("yscale", None)

    if ax is None:
        ax = plt.gca()

    ax.set_title("Convergence plot")
    ax.set_xlabel("Number of calls $n$")
    ax.set_ylabel(r"$\min f(x)$ after $n$ calls")
    ax.grid()

    if yscale is not None:
        ax.set_yscale(yscale)

    colors = plt.cm.viridis(np.linspace(0.25, 1.0, len(args)))

    for results, color in zip(args, colors):
        if isinstance(results, tuple):
            name, results = results
        else:
            name = None

        if isinstance(results, list):
            n_calls = len(results[0].x_iters)
            iterations = range(1, n_calls + 1)
            mins = [[np.min(r.func_vals[:i]) for i in iterations]
                    for r in results]

            for m in mins:
                ax.plot(iterations, m, c=color, alpha=0.2)

            ax.plot(iterations, np.mean(mins, axis=0), c=color,
                    marker=".", markersize=12, lw=2, label=name)
        else:
            n_calls = len(results.x_iters)
            mins = [np.min(results.func_vals[:i])
                    for i in range(1, n_calls + 1)]
            ax.plot(range(1, n_calls + 1), mins, c=color,
                    marker=".", markersize=12, lw=2, label=name)

    if true_minimum:
        ax.axhline(true_minimum, linestyle="--",
                   color="r", lw=1,
                   label="True minimum")

    if true_minimum or name:
        ax.legend(loc="best")

    return ax


def plot_res_gp(res_gp):
    """to plot results from the gaussian process optimization of scikit-optimize (skopt) package"""
    fig = plt.figure(figsize=(12, 6))
    my_plot_convergence(res_gp)
    plt.grid()
    plt.show()

    fig = plt.figure(figsize=(12, 6))
    plt.plot(res_gp.func_vals, 'bo-')  # 'b = blue, o = draw circles, - = draw lines between dots
    # plt.hold(True)
    # plt.scatter(range(len(res_gp.func_vals)), res_gp.func_vals)
    plt.ylabel(r'$f(x)$')
    plt.xlabel('Number of calls $n$')
    plt.xlim([0, len(res_gp.func_vals)])
    plt.show()


class MyOptimizeResult(object):
    def __init__(self, res_gp):
        super(MyOptimizeResult, self).__init__()
        self.x_iters = res_gp.x_iters
        self.func_vals = res_gp.func_vals
        self.best_params = res_gp.x
