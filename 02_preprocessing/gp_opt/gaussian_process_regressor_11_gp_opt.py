import pickle
import dill
from os import path
import numpy as np
from skopt.space.space import Integer, Real
from skopt import gp_minimize
import gc

from gaussian_process_price_prediction.gaussian_process_cluster_predictor import GaussianProcessPricePredictorForCluster
from gp_opt.my_optimize_result import MyOptimizeResult


class GaussianProcessPricePredictorGpOpt(object):
    NUM_JOBS = 1  # do NOT increase this unless you have lots of fast RAM and/or Virtual Memory (32GB or more!)

    def __init__(self, pairs_ts_npy_filename, cv_score_dict_npy_filename, res_gp_filename, bayes_opt_dir,
                 npz_sku_ids_group_kmeans, mobs_norm_path, price_history_csv, input_min_len, target_len, chosen_cluster,
                 random_state=None, plotter=None, verbose=False, **kwargs):
        super(GaussianProcessPricePredictorGpOpt, self).__init__()
        self.static_params = kwargs
        self.plotter = plotter
        self.random_state = random_state
        self.pairs_ts_filepath = bayes_opt_dir + '/' + pairs_ts_npy_filename + '_{:02d}.npy'.format(chosen_cluster)
        self.cv_score_dict_filepath = bayes_opt_dir + '/' + cv_score_dict_npy_filename + '_{:02d}.npy'.format(
            chosen_cluster)
        self.res_gp_filepath = bayes_opt_dir + '/{}_{:02d}.pickle'.format(res_gp_filename, chosen_cluster)

        self.gp_predictor = GaussianProcessPricePredictorForCluster(npz_sku_ids_group_kmeans=npz_sku_ids_group_kmeans,
                                                                    mobs_norm_path=mobs_norm_path,
                                                                    price_history_csv=price_history_csv,
                                                                    input_min_len=input_min_len,
                                                                    target_len=target_len)
        self.gp_predictor.prepare(chosen_cluster=chosen_cluster, verbose=verbose)
        self.verbose = verbose

    def run_opt(self, n_random_starts, n_calls):
        if path.isfile(self.res_gp_filepath):
            if self.verbose:
                print "there is already a path {}".format(self.res_gp_filepath)

            with open(self.res_gp_filepath) as fp:  # Python 3: open(..., 'rb')
                opt_res = pickle.load(fp)
        else:
            if self.verbose:
                print "there is NO file path {}".format(self.res_gp_filepath)

            res_gp = self.gpOptimization(n_random_starts=n_random_starts, n_calls=n_calls)
            opt_res = MyOptimizeResult(res_gp=res_gp)
            with open(self.res_gp_filepath, 'w') as fp:  # Python 3: open(..., 'wb')
                pickle.dump(opt_res, fp)

        return opt_res

    def objective(self, params):  # Here we define the metric we want to minimise
        params_str = "params: {}".format(params)
        if self.verbose:
            print 'INIT length_scale {}'.format(params)

        # try:
        cv_score, pairs_ts = self.get_or_calc(params=params)

        # save everytime in case it crashes
        self.__save_dictionary(filepath=self.pairs_ts_filepath, key=params, val=pairs_ts)
        self.__save_dictionary(filepath=self.cv_score_dict_filepath, key=params, val=cv_score)

        if self.plotter is not None:
            self.plotter(pairs_ts, label_text=params_str)

        if self.verbose:
            print "FINISHED for length scale: {}".format(params)

        gc.collect()  # clear unnecessary memory

        return cv_score  # minimize

    def get_or_calc(self, params):
        params = tuple(params)

        if path.isfile(self.cv_score_dict_filepath):
            cv_score_dict = np.load(self.cv_score_dict_filepath)[()]

            if params in cv_score_dict:
                if self.verbose:
                    print "params {} already exist".format(params)

                dic = np.load(self.pairs_ts_filepath)[()]
                assert params in dic, 'if you have created a cv score you must have saved the pairs already'

                cv_score, pairs = cv_score_dict[params], dic[params]
            else:
                if self.verbose:
                    print "params {} do NOT exist and need to be calculated".format(params)

                cv_score, pairs = self.calc(params=params)
        else:
            if self.verbose:
                print "cv score dictionary path {} does not exist and is going to be created after running for params {}".format(
                    self.cv_score_dict_filepath, params
                )

            cv_score, pairs = self.calc(params=params)

        return cv_score, pairs

    def calc(self, params):
        (length_scale,) = params

        dtw_mean, pairs = self.gp_predictor.train_validate(length_scale=length_scale, verbose=self.verbose,
                                                           **self.static_params)

        cv_score = dtw_mean

        return cv_score, pairs

    @staticmethod
    def __save_dictionary(filepath, key, val):
        if filepath is not None:
            dic = np.load(filepath)[()] if path.isfile(filepath) else dict()
            dic[tuple(key)] = val
            np.save(filepath, dic)

    def gpOptimization(self, n_random_starts, n_calls):
        # length_scale = Real(0.1, 3.0, prior='uniform')  # uniform or log-uniform
        length_scale = Real(0.01, 5.0, prior='uniform')  # uniform or log-uniform

        space = [length_scale]

        return gp_minimize(
            func=self.objective,  # function that we wish to minimise
            dimensions=space,  # the search space for the hyper-parameters
            # x0=x0, #inital values for the hyper-parameters
            n_calls=n_calls,  # number of times the function will be evaluated
            random_state=self.random_state,  # random seed
            n_random_starts=n_random_starts,  # before we start modelling the optimised function with a GP Regression
            # model, we want to try a few random choices for the hyper-parameters.
            kappa=3,  # trade-off between exploration vs. exploitation.
            n_jobs=self.NUM_JOBS
        )
