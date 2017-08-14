from collections import OrderedDict

import numpy as np
from os.path import isfile

from fastdtw import fastdtw
from pandas import read_csv
from csv import QUOTE_ALL

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from matplotlib import pyplot as plt
from helpers.py_helper import merge_dicts
from mobattrs_price_history_merger import MobAttrsPriceHistoryMerger
from price_history import PriceHistory
from skroutz_mobile import SkroutzMobile
from timeit import default_timer as timer


class GaussianProcessPricePredictorForCluster(object):
    DEFAULT_LENGTH_SCALE = 1.0  # http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html#sklearn.gaussian_process.kernels.RBF

    def __init__(self, npz_sku_ids_group_kmeans, mobs_norm_path, price_history_csv, input_min_len, target_len,
                 random_state=None):
        super(GaussianProcessPricePredictorForCluster, self).__init__()
        self.input_min_len = input_min_len
        self.target_len = target_len
        self.seq_min_len = input_min_len + target_len
        assert isfile(price_history_csv)
        self.price_history_csv = price_history_csv
        assert isfile(npz_sku_ids_group_kmeans)
        self.npz_sku_ids_group_kmeans = npz_sku_ids_group_kmeans
        assert isfile(mobs_norm_path)
        self.mobs_norm_path = mobs_norm_path
        self.train_df = None
        self.valid_df = None
        self.test_df = None
        self.df_cluster = None
        self.price_mean = None
        self.price_std = None
        self.random_state = random_state

    def train_test(self, length_scale, verbose=False):
        train_arr = self.train_df.append(self.valid_df).values

        if verbose:
            print train_arr.shape

        gpr = self.training(train_arr=train_arr, length_scale=length_scale, verbose=verbose)

        dtw_scores = []
        pairs = OrderedDict()
        for score, preds, targets, sku_id in self.get_dtw_scores(gpr=gpr, dataframe=self.test_df):
            dtw_scores.append(score)
            pairs[sku_id] = {
                'predictions': preds,
                'targets': targets
            }

        dtw_mean = np.mean(dtw_scores)
        if verbose:
            print dtw_mean

        del gpr  # clear memory

        return dtw_mean, pairs

    def train_validate(self, length_scale=DEFAULT_LENGTH_SCALE, n_restarts_optimizer=0, verbose=False):
        train_arr = self.train_df.values

        if verbose:
            print "starting training"
        start_timer = timer() if verbose else None

        gpr = self.training(train_arr=train_arr, length_scale=length_scale, verbose=verbose,
                            n_restarts_optimizer=n_restarts_optimizer)

        if verbose:
            end_timer = timer()
            print "elapsed time for training: {}".format(end_timer - start_timer)

        start_timer = timer() if verbose else None
        dtw_scores = []
        pairs = OrderedDict()
        for score, preds, targets, sku_id in self.get_dtw_scores(gpr=gpr, dataframe=self.valid_df):
            # if verbose:
            #     print "we have calculated the situation for sku {} and score is {}".format(sku_id, score)

            dtw_scores.append(score)
            pairs[sku_id] = {
                'predictions': preds,
                'targets': targets
            }

        dtw_mean = np.mean(dtw_scores)
        if verbose:
            end_timer = timer()
            print "elapsed time for dtw scores: {}".format(end_timer - start_timer)
            print dtw_mean

        del gpr  # clear memory

        return dtw_mean, pairs

    def training(self, train_arr, length_scale, n_restarts_optimizer=0, verbose=False):
        XX = train_arr[:, :MobAttrsPriceHistoryMerger.PRICE_IND]
        if verbose:
            print XX.shape

        yy = train_arr[:, MobAttrsPriceHistoryMerger.PRICE_IND]
        if verbose:
            print yy.shape

        kernel = RBF(length_scale=length_scale)
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=self.random_state,
                                       n_restarts_optimizer=n_restarts_optimizer)
        gpr.fit(XX, yy)

        return gpr

    def get_dtw_scores(self, gpr, dataframe):
        cluster_sku_ids = self.df_cluster.index
        for sku_id, inputs, targets in self.get_inputs_targets_per_sku_for_dataframe(cluster_sku_ids=cluster_sku_ids,
                                                                                     dataframe=dataframe):
            preds = gpr.predict(inputs)
            targets_denorm = self.denormalize_vector(targets, vec_mean=self.price_mean, vec_std=self.price_std)
            preds_denorm = self.denormalize_vector(preds, vec_mean=self.price_mean, vec_std=self.price_std)
            yield fastdtw(preds_denorm, targets_denorm)[0], preds_denorm, targets_denorm, sku_id

    @staticmethod
    def get_inputs_targets_per_sku_for_dataframe(cluster_sku_ids, dataframe):
        for cur_sku_id in cluster_sku_ids:
            if cur_sku_id in dataframe.index:
                vals = dataframe.loc[cur_sku_id].values
                xx = vals[:, :-1]
                tars = vals[:, -1]
                yield cur_sku_id, xx, tars
            else:
                continue

    def prepare(self, chosen_cluster, verbose=False):
        if self.__PREPARE_EXECUTED:
            raise Exception("prepare is allowed to be executed only once")

        """chosen_cluster is an index integer from 0 to ... number of clusters - 1"""
        df_cluster = self.loading_data(chosen_cluster=chosen_cluster, verbose=verbose)
        splitted_seqs = self.train_valid_test_split(df_cluster=df_cluster, verbose=verbose)
        if splitted_seqs is None:
            raise NotImplementedError  # TODO
        else:
            train_seqs, valid_seqs, test_seqs = splitted_seqs

            obj = MobAttrsPriceHistoryMerger(mobs_norm_path=self.mobs_norm_path,
                                             price_history_csv=self.price_history_csv)
            df_cluster_no_price = df_cluster.drop(labels=SkroutzMobile.PRICE_COLS, axis=1)

            metrics, price_mean, price_std = self.get_metrics(mobAttrsPriceHistoryMerger=obj,
                                                              df_cluster_no_price=df_cluster_no_price, verbose=verbose)

            train_df = self.generate_df_and_normalize(obj, metrics=metrics, df_cluster_no_price=df_cluster_no_price,
                                                      cur_seqs=train_seqs, verbose=verbose)

            valid_df = self.generate_df_and_normalize(obj, metrics=metrics, df_cluster_no_price=df_cluster_no_price,
                                                      cur_seqs=valid_seqs, verbose=verbose)

            test_df = self.generate_df_and_normalize(obj, metrics=metrics, df_cluster_no_price=df_cluster_no_price,
                                                     cur_seqs=test_seqs, verbose=verbose)

            self.train_df = train_df
            self.valid_df = valid_df
            self.test_df = test_df
            self.df_cluster = df_cluster
            self.price_mean = price_mean
            self.price_std = price_std

            self.__PREPARE_EXECUTED = True

    __PREPARE_EXECUTED = False

    def get_metrics(self, mobAttrsPriceHistoryMerger, df_cluster_no_price, verbose=False):
        if verbose:
            print df_cluster_no_price.shape

        df_full = mobAttrsPriceHistoryMerger.get_table(df=df_cluster_no_price, normalize_dates=False,
                                                       normalize_price=False)

        price_metrics = mobAttrsPriceHistoryMerger.get_normalize_price_metrics(df_full.values)
        if verbose:
            print price_metrics

        dic_metrics = mobAttrsPriceHistoryMerger.get_normalize_date_metrics(df_full.values)
        if verbose:
            print dic_metrics

        both_metrics = merge_dicts(price_metrics, dic_metrics)

        price_mean = price_metrics.values()[0][0]
        price_std = price_metrics.values()[0][1]

        return both_metrics, price_mean, price_std

    def generate_df_and_normalize(self, mobAttrsPriceHistoryMerger, metrics, df_cluster_no_price, cur_seqs, verbose):
        cur_df = mobAttrsPriceHistoryMerger.get_table(df=df_cluster_no_price,
                                                      seqs=cur_seqs, normalize_dates=False, normalize_price=False)
        return mobAttrsPriceHistoryMerger.normalize_df(cur_df, metrics, verbose=verbose)

    def train_valid_test_split(self, df_cluster, verbose=False):
        """
        We have a training set for our cluster with dynamic length and a test set with static length equal
        to our target length

        In addition we have completely separated the past (and present) from the future in order to not let
        our model be trained with any information that belong in the future.
        While we test how good the forecast is for the future
        """

        ph = PriceHistory(self.price_history_csv)
        seqs = ph.extractAllSequences()

        if verbose:
            print len(seqs)

        seqs_cluster = self.keepSeqsInCluster(seqs=seqs, df_cluster=df_cluster, verbose=verbose)

        seqs_long = self.keepLongEnoughSeqs(seqs_cluster=seqs_cluster, verbose=verbose)

        if len(seqs_long) == 0:
            return None
        else:
            full_dates = self.get_full_dates(seqs_long=seqs_long, verbose=verbose)

            final_threshold_date = self.find_threshold(full_date_list=full_dates, target_length=self.target_len,
                                                       sequences=seqs_long)
            if verbose:
                print final_threshold_date

            full_dates = np.array(full_dates)

            train_dates = full_dates[full_dates < final_threshold_date]
            if verbose:
                print len(train_dates)

            train_seqs_with_nans = [seq[train_dates] for seq in seqs_long]
            if verbose:
                print len(train_seqs_with_nans)

            train_seqs = [seq[seq == seq] for seq in train_seqs_with_nans]

            index_threshold = np.argwhere(full_dates == final_threshold_date).flatten()
            assert len(index_threshold) == 1
            index_threshold = index_threshold[0]
            if verbose:
                print index_threshold

            end_target_ind = index_threshold + self.target_len
            if verbose:
                print end_target_ind

            target_end_date = full_dates[end_target_ind]
            if verbose:
                print target_end_date

            # BEAWARE to put the numpy array first on an operation in order for the numpy to be triggered
            valid_dates = full_dates[np.logical_and(full_dates >= final_threshold_date, full_dates < target_end_date)]
            if verbose:
                print len(valid_dates)

            valid_seqs_with_nans = [seq[valid_dates] for seq in seqs_long]
            valid_seqs_var_len = [seq[seq == seq] for seq in valid_seqs_with_nans]

            # assert np.all([len(seq) == self.target_len for seq in valid_seqs]), repr(
            #     [len(seq) for seq in valid_seqs]
            # )
            valid_seqs = [seq for seq in valid_seqs_var_len if len(seq) == self.target_len]

            test_seqs = self.get_test_seqs(full_dates=full_dates, target_end_date=target_end_date,
                                           seqs_long=seqs_long, verbose=verbose)

            return train_seqs, valid_seqs, test_seqs

    def get_test_seqs(self, full_dates, target_end_date, seqs_long, verbose=False):
        test_dates = full_dates[full_dates >= target_end_date][:self.target_len]
        test_seqs_with_nans = [seq[test_dates] for seq in seqs_long]
        test_seqs_varlen = [seq[seq == seq] for seq in test_seqs_with_nans]
        test_seqs = [seq for seq in test_seqs_varlen if len(seq) >= self.target_len]
        if verbose:
            print len(test_seqs), ",".join([str(len(seq)) for seq in test_seqs])
        return test_seqs

    def find_threshold(self, full_date_list, target_length, sequences):
        """
        pick 30 days before the end of the union of the dates from our kept sequences and start going back in time
        until you find a date where ALL the sequences have as target a sequence of length 30 or more

        Note that this threshold should be unique since we do NOT want to give the model any information about the
        future. So we are being strict that we train for the past (and present day) and we are testing on our
        predictions on an unknown future
        """
        for ii in range(len(full_date_list) - target_length, 0, -1):
            cur_threshold_date = full_date_list[ii]
            if self.haveAllSeqsEnoughTarget(sequences, cur_threshold_date, target_length):
                return cur_threshold_date

        # one idea is to drop short sequences..
        # another idea is to drop old sequences...
        # anyway we will deal with this case if and when we get this exception
        raise Exception("no common threshold date found")

    @staticmethod
    def haveAllSeqsEnoughTarget(sequences, threshold_date, target_length):
        target_seqs = [seq[seq.index >= threshold_date] for seq in sequences]
        return np.all([len(target_seq) >= target_length * 2 for target_seq in target_seqs])

    def get_full_dates(self, seqs_long, verbose=False):
        full_dates = set()

        for seq in seqs_long:
            full_dates = full_dates.union(seq.index.values)

        full_dates = sorted(full_dates)

        if verbose:
            print full_dates[:5]

        return full_dates

    def keepLongEnoughSeqs(self, seqs_cluster, verbose):
        seqs_long = [seq for seq in seqs_cluster if len(seq) >= self.seq_min_len]

        if verbose:
            print len(seqs_long)

        return seqs_long

    def keepSeqsInCluster(self, seqs, df_cluster, verbose=False):
        seqs_cluster = [seq for seq in seqs if seq.name in df_cluster.index]
        if verbose:
            print len(seqs_cluster)
        return seqs_cluster

    def loading_data(self, chosen_cluster, verbose=False):
        """chosen_cluster is an index integer from 0 to ... number of clusters - 1"""
        sku_id_groups = np.load(self.npz_sku_ids_group_kmeans)

        if verbose:
            for key, val in sku_id_groups.iteritems():
                print key, ",", val.shape

        chosen_cluster = str(chosen_cluster)  # str because this is how values are stored as keys in npz files

        df = read_csv(self.mobs_norm_path, index_col=0, encoding='utf-8', quoting=QUOTE_ALL)
        assert np.all(np.logical_not(np.isnan(df.values.flatten()))), "the dataframe should not contain any nan values"

        if verbose:
            print df.shape

        cluster_sku_ids = set(sku_id_groups[chosen_cluster]).intersection(df.index)

        if verbose:
            print len(cluster_sku_ids)

        df_cluster = df.loc[cluster_sku_ids]

        if verbose:
            print df_cluster.shape

        return df_cluster

    @staticmethod
    def denormalize_vector(vector, vec_mean, vec_std):
        return (vector * vec_std) + vec_mean

    def plot_random_prediction(self, gpr):
        cluster_sku_ids = self.df_cluster.index
        cur_sku_id = list(cluster_sku_ids)[np.random.randint(len(cluster_sku_ids))]
        vals = self.test_df.loc[cur_sku_id].values
        xx = vals[:, :-1]

        tars = self.denormalize_vector(vals[:, -1], vec_mean=self.price_mean, vec_std=self.price_std)
        preds = self.denormalize_vector(gpr.predict(xx), vec_mean=self.price_mean, vec_std=self.price_std)

        plt.figure()
        plt.plot(tars, 'r-', label='targets')
        plt.plot(preds, 'b-', label='predictions')
        plt.legend()

        # VERIFICATION THAT THE RECONSTRUCTION OF THE TARGETS WORKS OK
        # for sku_id, inputs, targets in get_inputs_targets_per_sku():
        #     assert len(inputs) == target_len
        #     targets_denorm = denormalize_prices(targets, price_metrics.values()[0])
        #     # print targets_denorm
        #     other_tars = get_target_seq_by_sku_id(target_seqs, sku_id=sku_id)
        #     assert np.allclose(other_tars.values, targets_denorm)
