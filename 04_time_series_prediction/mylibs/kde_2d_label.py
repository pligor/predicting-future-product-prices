import seaborn as sns
from matplotlib import cm
import numpy as np

def kde_2d_label(X_2d, y, alpha=0.5):
    """Visualuse a 2D embedding with corresponding labels.

    X_2d : ndarray, shape (n_samples,2)
        Low-dimensional feature representation.

    y : ndarray, shape (n_samples,)
        Labels corresponding to the entries in X_2d.

    alpha : float
        Transparency for scatter plot.
    """
    # colors = sns.color_palette(n_colors=targets.size)
    pool_of_cmaps = [cm.Blues, cm.Greens, cm.Reds, cm.Greys, cm.Oranges, cm.Purples]
    # http://stackoverflow.com/questions/37902459/how-do-i-use-seaborns-color-palette-as-a-colormap-in-matplotlib
    targets = np.unique(y)

    cmaps = []
    for i in range(len(targets)):
        cmaps.append(pool_of_cmaps[i % len(pool_of_cmaps)])

    for cmap, target in zip(cmaps, targets):
        # ax = sns.kdeplot(X_2d[y==target], cmap=cmap, alpha=alpha)
        # the above and below are identical actually
        data = X_2d[y == target]
        sns.kdeplot(data[:, 0], data[:, 1], cmap=cmap, alpha=alpha)