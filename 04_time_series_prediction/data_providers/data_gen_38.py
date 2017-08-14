from collections import OrderedDict

import numpy as np
from os.path import isfile

from fastdtw import fastdtw
from pandas import read_csv
from csv import QUOTE_ALL

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from matplotlib import pyplot as plt
from mylibs.py_helper import merge_dicts
from price_history import PriceHistory
from skroutz_mobile import SkroutzMobile
from timeit import default_timer as timer


class DatasetGenerator(object):
    def __init__(self, npz_sku_ids_group_kmeans, price_history_csv, input_min_len, target_len, do_valid_split = True,
                 random_state=None):


        super(DatasetGenerator, self).__init__()
        self.input_min_len = input_min_len
        self.target_len = target_len
        self.seq_min_len = input_min_len + (target_len * 2 if do_valid_split else target_len)
        assert isfile(price_history_csv)
        self.price_history_csv = price_history_csv
        assert isfile(npz_sku_ids_group_kmeans)
        self.npz_sku_ids_group_kmeans = npz_sku_ids_group_kmeans
        self.train_df = None
        self.valid_df = None
        self.test_df = None
        self.df_cluster = None
        self.price_mean = None
        self.price_std = None
        self.random_state = random_state
        self.do_valid_split = do_valid_split

    @staticmethod
    def get_inputs_targets_per_sku_for_dataframe(cluster_sku_ids, dataframe):
        for cur_sku_id in cluster_sku_ids:
            vals = dataframe.loc[cur_sku_id].values
            xx = vals[:, :-1]
            tars = vals[:, -1]
            yield cur_sku_id, xx, tars

    def prepare(self, chosen_cluster, verbose=False):
        if self.__PREPARE_EXECUTED:
            raise Exception("prepare is allowed to be executed only once")

        """chosen_cluster is an index integer from 0 to ... number of clusters - 1"""
        cluster_sku_ids = self.get_cluster_sku_ids(chosen_cluster=chosen_cluster, verbose=verbose)

        if verbose:
            print cluster_sku_ids

        splitted_seqs = self.train_valid_test_split(cluster_sku_ids=cluster_sku_ids, verbose=verbose)

        return splitted_seqs
        if splitted_seqs is None:
            raise NotImplementedError  # TODO
        else:
            train_seqs, valid_seqs, test_seqs = splitted_seqs

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

    def train_valid_test_split(self, cluster_sku_ids, verbose=False):
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

        seqs_cluster = self.keepSeqsInCluster(seqs=seqs, cluster_sku_ids=cluster_sku_ids, verbose=verbose)

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
            valid_seqs = [seq[seq == seq] for seq in valid_seqs_with_nans]
            assert np.all([len(seq) == self.target_len for seq in valid_seqs])

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

    def haveAllSeqsEnoughTarget(self, sequences, threshold_date, target_length):
        target_seqs = [seq[seq.index >= threshold_date] for seq in sequences]
        return np.all([len(target_seq) >= (target_length * 2 if self.do_valid_split else target_length) for target_seq in target_seqs])

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

    def keepSeqsInCluster(self, seqs, cluster_sku_ids, verbose=False):
        seqs_cluster = [seq for seq in seqs if seq.name in cluster_sku_ids]
        if verbose:
            print len(seqs_cluster)
        return seqs_cluster

    def get_cluster_sku_ids(self, chosen_cluster, verbose=False):
        """chosen_cluster is an index integer from 0 to ... number of clusters - 1"""
        sku_id_groups = np.load(self.npz_sku_ids_group_kmeans)

        return sku_id_groups[str(chosen_cluster)] # str because this is how values are stored as keys in npz files

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
