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
from skroutz_mobile import SkroutzMobile
from sklearn.ensemble import IsolationForest


def printgr(obj):
    print repr(obj).decode('unicode-escape')


class BasicPreprocessing(object):
    HEIGHT_ORIG_COL = 'height'
    WIDTH_ORIG_COL = 'width'
    DEPTH_ORIG_COL = 'depth'
    HW_COL = 'height_width_area_norm'
    HD_COL = 'height_depth_area_norm'
    WD_COL = 'width_depth_area_norm'
    VOLUME_COL = 'volume_norm'

    VER_RES_ORIG_COL = 'vertical_resolution'
    HOR_RES_ORIG_COL = 'horizontal_resolution'
    SCREEN_RES_COL = 'screen_res_norm'

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

    def __process_dimensions(self, heights, widths, depths):
        heights_widths_normalized = np.sqrt(heights * widths)
        heights_depths_normalized = np.sqrt(heights * depths)
        widths_depths_normalized = np.sqrt(widths * depths)

        volume_normalized = np.power(heights * widths * depths, 1. / 3.)

        return {
            self.HW_COL: heights_widths_normalized,
            self.HD_COL: heights_depths_normalized,
            self.WD_COL: widths_depths_normalized,
            self.VOLUME_COL: volume_normalized,
        }

    def __process_screen_resolution(self, vertical_res, horizontal_res):
        screen_res_normalized = np.sqrt(vertical_res * horizontal_res)
        return {
            self.SCREEN_RES_COL: screen_res_normalized
        }

    def process_screen_resolution(self, df):
        return self.__process_screen_resolution(vertical_res=df[self.VER_RES_ORIG_COL],
                                                horizontal_res=df[self.HOR_RES_ORIG_COL])

    def process_dimensions(self, df):
        return self.__process_dimensions(heights=df[self.HEIGHT_ORIG_COL],
                                         widths=df[self.WIDTH_ORIG_COL],
                                         depths=df[self.DEPTH_ORIG_COL])

    def introducing_extra_cols(self, csv_in="../mobiles_imputed.csv",
                               csv_out="../mobiles_01_preprocessed.csv"):
        df = pd.read_csv(csv_in, index_col=0, encoding='utf-8', quoting=csv.QUOTE_ALL)

        col_dic = dict(self.process_screen_resolution(df=df).items() +
                       self.process_dimensions(df=df).items())

        for key, val in col_dic.iteritems():
            df = pd.concat((df, val.to_frame(name=key)), axis=1)

        df.to_csv(csv_out, encoding='utf-8', quoting=csv.QUOTE_ALL)

    def getOutlierCols(self, all_columns):
        return list(
            set(all_columns).difference(SkroutzMobile.BINARY_COLS + self.OUTLIERS_NEGLECT_COLS))  # for now all cols

    def removeOutliersWithQuartiles(self, df, k=3, orig_df=None):
        """k=3 is enough to remove the apple iphone7 which is not good,
        it suggests a contamination of ~7%"""
        # print len(outlier_columns)

        outlier_columns = self.getOutlierCols(all_columns=df.columns)

        types = df[outlier_columns].dtypes
        for col in outlier_columns:
            types[col] = MyOutliers().getLooseBoundaries(df[col], k=k)
        bounds_k3 = types.copy()

        outliersCount = MyOutliers.countOutliersDataPoints(df, bounds_k3)

        # print bounds_k3
        print outliersCount[outliersCount != 0]

        df_new, fails = MyOutliers().removeOutliers(data=df, bounds=bounds_k3)

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
        qr_survivors, qr_fails = self.removeOutliersWithQuartiles(df=df, k=6)
        print "REMOVING OUTLIERS"

        print len(qr_fails)

        if_survivors, if_fails = self.removeOutliersIsolationForest(contamination=0.01, df=df,
                                                                    random_state=random_state, n_jobs=n_jobs)
        print len(if_fails)

        fails_index = set(if_fails.index).union(qr_fails.index)
        survivors_index = set(qr_survivors.index).intersection(if_survivors.index)

        # qr_fails = orig_df.loc[df_qr_fails.index, 'display_name']
        print len(fails_index)

        survivors = orig_df.loc[survivors_index]
        # print len(survivors)
        if save:
            survivors.to_csv(csv_out, encoding='utf-8', quoting=csv.QUOTE_ALL)

        return survivors


# TODO remove outliers based on distribution shapes
if __name__ == "__main__":
    seed = 16011984
    random_state = np.random.RandomState(seed=seed)
    n_jobs = 1
    # print set([4,2]).union([4, 1])
    # exit(1)

    bp = BasicPreprocessing()
    # bp.introducing_extra_cols()
    csv_in = "../mobiles_01_preprocessed.csv"

    orig_df = pd.read_csv(csv_in, index_col=0, encoding='utf-8', quoting=csv.QUOTE_ALL)

    df = orig_df.drop(labels=SkroutzMobile.TEMP_DROP_COLS, axis=1)
    print len(df)
    # print [col for col in df.columns if "price" in col]

    bp.removeOutliers(df=df, orig_df=orig_df, random_state=random_state, n_jobs=n_jobs, save=False)
    #note that we are removing only a small percentage of 2% of outliers,
    # this includes apple iphone 7 plus with 256GB, which is expected since it is too expensive

    print "DONE"
    os.system('play --no-show-progress --null --channels 1 synth {} sine {}'.format(0.5, 800))
