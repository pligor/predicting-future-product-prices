from os.path import isdir

from models.model_21_price_history_seq2seq_dyn_dec_ins import PriceHistorySeq2SeqDynDecIns
import pickle
import dill
from os import path, remove
import numpy as np
from skopt.space.space import Integer, Real, Categorical
from skopt import gp_minimize
import tensorflow as tf

from mylibs.jupyter_notebook_helper import MyOptimizeResult


class PriceHistory26GpOpt(object):
    NUM_GPUS = 1

    def __init__(self, model, stats_npy_filename, cv_score_dict_npy_filename, res_gp_filename, random_state=None,
                 plotter=None, bayes_opt_dir='../data/bayes_opt', **kwargs):
        super(PriceHistory26GpOpt, self).__init__()
        assert isdir(bayes_opt_dir)
        self.model = model
        self.static_params = kwargs
        self.plotter = plotter
        self.random_state = random_state
        self.stats_filepath = bayes_opt_dir + '/' + stats_npy_filename + '.npy'
        self.cv_score_dict_filepath = bayes_opt_dir + '/' + cv_score_dict_npy_filename + '.npy'
        self.res_gp_filepath = bayes_opt_dir + '/{}.pickle'.format(res_gp_filename)

    def run_opt(self, n_random_starts, n_calls):
        if path.isfile(self.res_gp_filepath):
            with open(self.res_gp_filepath) as fp:  # Python 3: open(..., 'rb')
                opt_res = pickle.load(fp)
        else:
            res_gp = self.gpOptimization(n_random_starts=n_random_starts, n_calls=n_calls)
            opt_res = MyOptimizeResult(res_gp=res_gp)
            with open(self.res_gp_filepath, 'w') as fp:  # Python 3: open(..., 'wb')
                pickle.dump(opt_res, fp)

        return opt_res

    def objective(self, params):  # Here we define the metric we want to minimise
        params_str = "params: {}".format(params)
        print params_str
        cv_score, stats_list = self.get_or_calc(params=params)

        # save everytime in case it crashes
        self.__save_dictionary(filepath=self.stats_filepath, key=params, val=stats_list)
        self.__save_dictionary(filepath=self.cv_score_dict_filepath, key=params, val=cv_score)

        if self.plotter is not None:
            self.plotter(stats_list=stats_list, label_text=params_str)

        return cv_score  # minimize validation error

    def get_or_calc(self, params):
        params = tuple(params)

        if path.isfile(self.cv_score_dict_filepath):
            cv_score_dict = np.load(self.cv_score_dict_filepath)[()]

            if params in cv_score_dict:
                stats_dic = np.load(self.stats_filepath)[()]
                assert params in stats_dic, 'if you have created a cv score you must have saved the stats list before'

                cv_score, stats_list = cv_score_dict[params], stats_dic[params]
            else:
                cv_score, stats_list = self.calc(params=params)
        else:
            cv_score, stats_list = self.calc(params=params)

        return cv_score, stats_list

    def calc(self, params):
        (num_units, lamda2, keep_prob_input, learning_rate, rnn_hidden_dim, mobile_attrs_dim) = params

        cv_score, stats_list = self.model.get_cross_validation_score(
            # npz_path=self.static_params['npz_train'],
            # epochs=self.static_params['epochs'],
            # batch_size=self.static_params['batch_size'],
            # input_len=self.static_params['input_len'],
            # target_len=self.static_params['target_len'],
            # n_splits=self.static_params['n_splits'],

            num_units=num_units,
            lamda2=lamda2,
            keep_prob_input=keep_prob_input,
            learning_rate=learning_rate,
            rnn_hidden_dim=rnn_hidden_dim,
            mobile_attrs_dim=mobile_attrs_dim,

            # DO NOT TEST
            decoder_first_input=PriceHistorySeq2SeqDynDecIns.DECODER_FIRST_INPUT.ZEROS,
            batch_norm_enabled=True,

            **self.static_params
        )

        return cv_score, stats_list

    @staticmethod
    def __save_dictionary(filepath, key, val):
        if filepath is not None:
            stats_dic = np.load(filepath)[()] if path.isfile(filepath) else dict()
            stats_dic[tuple(key)] = val
            np.save(filepath, stats_dic)

    # def __clear_previously_saved_files(self):
    #     #filepaths = [self.stats_filepath, self.cv_score_dict_filepath]
    #     filepaths = [self.stats_filepath,]
    #     for filepath in filepaths:
    #         if path.isfile(filepath):
    #             remove(self.stats_filepath)  # delete previously saved file

    def gpOptimization(self, n_random_starts, n_calls):
        # self.__clear_previously_saved_files()

        # here we will exploit the information from the previous experiment to calibrate what we think are the best parameters
        # best_params = [500,  #50 was obviously small so we are going to range it from 300 to 700
        #                tf.nn.tanh,  #activation we are not going to try and guess via gp opt, but just use this one
        #                0.0001,  #since we had as optimal the smallest one we are going to try and allow also smaller values
        #                0.62488034788862112,
        #                0.001]

        num_units = Integer(300, 700)
        lamda2 = Real(1e-5, 1e0, prior='log-uniform')  # uniform or log-uniform
        keep_prob_input = Real(0.2, 1.0, prior='uniform')  # uniform or log-uniform
        learning_rate = Real(1e-6, 1e-2, prior='log-uniform')  # uniform or log-uniform
        rnn_hidden_dim = Integer(100, 500)  # very rough guess
        mobile_attrs_dim = Integer(200, 600)  # very rough guess

        space = [num_units, lamda2, keep_prob_input, learning_rate, rnn_hidden_dim, mobile_attrs_dim]

        return gp_minimize(
            func=self.objective,  # function that we wish to minimise
            dimensions=space,  # the search space for the hyper-parameters
            # x0=x0, #inital values for the hyper-parameters
            n_calls=n_calls,  # number of times the function will be evaluated
            random_state=self.random_state,  # random seed
            n_random_starts=n_random_starts,  # before we start modelling the optimised function with a GP Regression
            # model, we want to try a few random choices for the hyper-parameters.
            kappa=1.9,  # trade-off between exploration vs. exploitation.
            n_jobs=self.NUM_GPUS
        )
