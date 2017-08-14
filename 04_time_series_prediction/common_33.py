from os.path import isfile
import numpy as np


def get_or_run_nn(callback, filename, nn_runs_folder='../data/nn_runs'):
    """returns: dyn_stats, preds_dict"""
    npz_path = nn_runs_folder + '/{}.npz'.format(filename)

    if isfile(npz_path):
        dic = np.load(npz_path)
        return dic['dyn_stats'][()], dic['preds_dict'][()], dic['targets'][()], dic['twods'][()]
    else:
        dyn_stats, preds_dict, targets, twods = callback()
        np.savez(npz_path, dyn_stats=dyn_stats, preds_dict=preds_dict, targets=targets, twods=twods)
        return dyn_stats, preds_dict, targets, twods
