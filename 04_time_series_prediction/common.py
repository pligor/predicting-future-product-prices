from os.path import isfile
import numpy as np


def get_or_run_nn(callback, filename, nn_runs_folder='../data/nn_runs'):
    """returns: dyn_stats, preds_dict"""
    filepath = nn_runs_folder + '/{}.npz'.format(filename)

    if isfile(filepath):
        dic = np.load(filepath)
        if 'targets' in dic.keys():
            return dic['dyn_stats'][()], dic['preds_dict'][()], dic['targets'][()]
        else:
            return dic['dyn_stats'][()], dic['preds_dict'][()]
    else:
        arr = callback()
        if len(arr) > 2:
            dyn_stats, preds_dict, targets = arr
            np.savez(filepath, dyn_stats=dyn_stats, preds_dict=preds_dict, targets=targets)
            return dyn_stats, preds_dict, targets
        else:
            dyn_stats, preds_dict = arr
            np.savez(filepath, dyn_stats=dyn_stats, preds_dict=preds_dict)
            return dyn_stats, preds_dict
