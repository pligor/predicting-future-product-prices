# -*- coding: UTF-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import sys
import math
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import re
import os
import csv
from helpers.outliers import MyOutliers
from sklearn.ensemble import IsolationForest
from price_history import PriceHistory


def printgr(obj):
    print repr(obj).decode('unicode-escape')


class PreprocessPriceHistory(object):
    # """
    # reviews_count             21   Well this is expected
    # weight                     2   Too heavy of cellphones
    # height                     1   Too high
    # price_min                  2   Too expensive should not be left out
    # volume_norm                4   This is generated but it could be filtered out for very large phones
    # diagonal_size              1   diagonal size could also be very large
    # storage                    4   having too large of a storage could be abnormal as well
    # count_empty_fields        11   counting empty fields
    # width_depth_area_norm      3
    # height_depth_area_norm     7
    # speaking_autonomy          5
    # stanby_autonomy            2
    # battery_capacity           1
    # depth                     10
    # shop_count                 8
    # """
    OUTLIERS_NEGLECT_COLS = ['reviews_count', 'price_max', 'price_min']  # prices could be targets

    def getOutlierCols(self, all_columns):
        return list(set(all_columns).difference(PriceHistory.SPECIAL_COLS))

    # getOutliersIndices

    @staticmethod
    def getBounds(df, kk, outlier_columns):
        types = df[outlier_columns].dtypes
        for col in outlier_columns:
            cur_data_col = df[col]
            cur_data_col_no_zeros = cur_data_col[cur_data_col > 0]
            types[col] = MyOutliers().getLooseBoundaries(cur_data_col_no_zeros, k=kk)
        return types.copy()

    @staticmethod
    def countOutliers(df, bounds):
        outliersCount = MyOutliers.countOutliersDataPoints(
            df, bounds, filtering=lambda arr: arr > 0)

        return outliersCount[outliersCount != 0]

    def removeOutliersWithQuartiles(self, df, kk=3, orig_df=None, verbose=False):
        """k=3 is enough to remove the apple iphone7 which is not good,
        it suggests a contamination of ~7%"""
        # print len(outlier_columns)
        outlier_cols = self.getOutlierCols(all_columns=df.columns)

        bounds = self.getBounds(df=df, kk=kk, outlier_columns=outlier_cols)

        df_new, fails = MyOutliers().removeOutliers(data=df, bounds=bounds)

        if orig_df is not None:
            print "OUTLIERS via Quartiles"
            print orig_df.loc[fails.index, 'display_name']
        print len(df_new)
        print len(df_new) / len(df)

        return df_new, fails

    def removeOutliersIsolationForest(self, contamination, df, random_state, n_jobs, orig_df=None):
        isof = IsolationForest(  # n_estimators=100, #max_samples=60,
            contamination=contamination, random_state=random_state, n_jobs=n_jobs)
        outlier_columns = self.getOutlierCols(all_columns=df.columns)
        inlier_columns = list(set(df.columns).difference(outlier_columns))
        isof.fit(df[inlier_columns])

        preds = isof.predict(df[inlier_columns])
        # +1 is inlier while -1 is outlier
        fails = df.index[preds == -1]
        survivors = df.index[preds == 1]

        if orig_df is not None:
            print orig_df.loc[fails, 'display_name']

        return df.loc[survivors], df.loc[fails]

    def removeOutliers(self, df, orig_df, random_state, n_jobs, csv_out="../mobiles_02_no_outliers.csv", save=False):
        qr_survivors, qr_fails = self.removeOutliersWithQuartiles(df=df, kk=6)

        if_survivors, if_fails = self.removeOutliersIsolationForest(contamination=0.01, df=df,
                                                                    random_state=random_state, n_jobs=n_jobs)

        fails_index = set(if_fails.index).union(qr_fails.index)
        survivors_index = set(qr_survivors.index).intersection(if_survivors.index)

        # qr_fails = orig_df.loc[df_qr_fails.index, 'display_name']

        survivors = orig_df.loc[survivors_index]
        # print len(survivors)
        if save:
            survivors.to_csv(csv_out, encoding='utf-8', quoting=csv.QUOTE_ALL)

        return survivors

    @staticmethod
    def removeSpikes(df, outliers_per_sku):
        """
        :param outliers_per_sku:
        8874019                                              [71]
        11860092                                              [0]
        9633962                                              [96]
        10909904                                 [32, 33, 88, 89]
        8012281                                      [0, 1, 2, 3]
        10922599                                     [56, 57, 58]
        9333571     [342, 343, 344, 345, 346, 347, 348, 349, 350]
        10468270                                              [0]
        10001441                                       [201, 202]
        10019997                                             [72]
        9473245                                             [144]
        10646927                                            [180]
        9332994                    [180, 181, 183, 184, 185, 186]
        7945834                                      [0, 1, 2, 3]
        11775269                                              [0]
        11435749                                              [0]
        8130418                                   [329, 330, 331]
        """

        dataframe = df.copy()
        for sku_id, outlier_inds in outliers_per_sku.iteritems():
            # print df.loc[sku_id]
            series_with_zeros = df.loc[sku_id]
            series = series_with_zeros[series_with_zeros > 0]

            for outlier_ind in outlier_inds:
                left_value = 0
                for ii in xrange(outlier_ind - 1, 0, -1):
                    if ii in outlier_inds:
                        continue
                    else:
                        cur_val = series.iloc[ii]
                        if cur_val > 0:
                            left_value = cur_val
                            break

                right_value = 0
                for jj in xrange(outlier_ind + 1, len(series)):
                    if jj in outlier_inds:
                        continue
                    else:
                        cur_val = series.iloc[jj]
                        if cur_val > 0:
                            right_value = cur_val
                            break

                if left_value == 0 and right_value > 0:
                    left_value = right_value
                elif left_value > 0 and right_value == 0:
                    right_value = left_value
                elif left_value == 0 and right_value == 0:
                    raise Exception("extreme case")
                else:
                    pass

                mean_val = (left_value + right_value) / 2.
                target = series.iloc[outlier_ind:outlier_ind + 1].index[0]
                dataframe.loc[sku_id, target] = mean_val

        return dataframe

    @staticmethod
    def createSeqStartColumnOutOfOriginalPrices(df):
        seq_start = pd.DataFrame([np.argwhere(row > 0)[0][0] for ind, row in df.iterrows()], columns=['seq_start'],
                                 index=df.index)
        return pd.concat((df, seq_start), axis=1)

    @staticmethod
    def convertToPriceDifferences(df):
        """practically take each price history and convert it so that it becomes differences from the original price.
        Effectively this makes each sequence start from zero and then be either positive or negative according to the way
        the price increases or decreases correspondingly (mostly decreases)"""
        seq_start = df['seq_start']

        df_norm = df.copy(deep=True)
        for ind, row in df.iterrows():
            cur_pos = seq_start.loc[ind]
            start_price = row.iloc[cur_pos]
            cur_seq_len = row['sequence_length']
            end_pos = int(cur_pos + cur_seq_len - 1)
            cur_slice = slice(cur_pos, end_pos + 1)
            row_norm = row.copy(deep=True)
            row_norm.iloc[cur_slice] = [(item - start_price) for item in row.iloc[cur_slice]]
            df_norm.loc[ind] = row_norm

        return df_norm

    @staticmethod
    def keepOnlyPriceHistoriesWithMostFrequentLength(csv_in):
        dataframe = pd.read_csv(csv_in, index_col=0, encoding='utf-8', quoting=csv.QUOTE_ALL)
        ph = PriceHistory(csv_filepath_or_df=csv_in)
        seqs = ph.extractAllSequences()
        count_lens = dataframe.groupby(by='sequence_length')['sequence_length'].count()
        most_freq_len = int(np.argmax(count_lens))
        df_most_fr = dataframe[dataframe['sequence_length'] == most_freq_len]

        inds = df_most_fr.index
        seqs_fixed_width = np.array([seq.values for seq in seqs if len(seq) == most_freq_len])

        seqs_df = pd.DataFrame(data=seqs_fixed_width, index=inds)

        sequence_length = pd.Series(data=np.repeat(most_freq_len, len(seqs_df)), index=inds,
                                    name='sequence_length')

        seq_start = pd.Series(data=np.repeat(0, len(seqs_df)), index=inds, name='seq_start')

        df_fixed_width = pd.concat((seqs_df, sequence_length, seq_start), axis=1)

        return df_fixed_width


# TODO remove outliers based on distribution shapes
if __name__ == "__main__":
    seed = 16011984
    random_state = np.random.RandomState(seed=seed)
    n_jobs = 1

    pph = PreprocessPriceHistory()
    csv_in = "../price_history_00.csv"

    orig_df = pd.read_csv(csv_in, index_col=0, encoding='utf-8', quoting=csv.QUOTE_ALL)
    print orig_df.shape

    df = orig_df.drop(labels=PriceHistory.SPECIAL_COLS, axis=1)
    print df.shape
    # print [col for col in df.columns if "price" in col]

    # flat_df = pd.DataFrame(data=df.values.flatten(), columns=["all_prices"])

    # pph.removeOutliersWithQuartiles(df=df.T, kk=12, verbose=True)
    # print len(flat_df) / counter[0]

    # bp.removeOutliers(df=df, orig_df=orig_df, random_state=random_state, n_jobs=n_jobs, save=True)
    # note that we are removing only a small percentage of 2% of outliers,

    print "DONE"
    # os.system('play --no-show-progress --null --channels 1 synth {} sine {}'.format(0.5, 800))
