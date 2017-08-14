# -*- coding: UTF-8 -*-
from __future__ import division

import sys
import unirest
import json
from time import sleep
import pickle
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import os
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeSVD, MICE, \
    MatrixFactorization, BiScaler
import csv

CSV_FILENAME = "../mobiles_basic_preprocess.csv"
CSV_OUT_FILEPATH = "../mobiles_imputed.csv"
TEMP_DROP_COLS = ['main_image', 'screen_type', 'display_name']
FOREVER_DROP_COLS = [
    'future',  # is always the same, redundant feature
    'sar',  # too many values are missing

    # remove them now include them later
    # 'height_width_area_norm',
    # 'height_depth_area_norm',
    # 'width_depth_area_norm',
    # 'volume_norm',
    # 'screen_res_norm',
]
DROP_COLS = TEMP_DROP_COLS + FOREVER_DROP_COLS


class SkroutzImputation(object):
    """https://github.com/hammerlab/fancyimpute"""

    @staticmethod
    def generateEmptyRandomSeries(cols):
        dic = dict()
        for col in cols:
            dic[col] = np.NaN
        return pd.Series(data=dic, name=str(np.random.rand()))

    @staticmethod
    def buildComparisonDf(XX, X_filled, disp_name_col, null_mask_col):
        df = pd.DataFrame(data=X_filled, index=XX.index, columns=XX.columns)
        df.insert(loc=0, column='disp_name', value=disp_name_col)
        return df.loc[null_mask_col]

    def imputing_column(self, col_id, XX, disp_name_col, filleds):
        cols = XX.columns

        col_name = cols[col_id]
        print col_name
        print

        null_mask = XX.isnull()[col_name]

        only_nulls = XX[col_name][null_mask]
        # print len(only_nulls)
        # print only_nulls
        # print

        # print X_filled_knn.shape
        # print type(X_filled_knn)
        df_knn_3 = SkroutzImputation.buildComparisonDf(XX=XX, X_filled=filleds[0],
                                                       disp_name_col=disp_name_col,
                                                       null_mask_col=null_mask)

        df_knn_5 = SkroutzImputation.buildComparisonDf(XX=XX, X_filled=filleds[1],
                                                       disp_name_col=disp_name_col,
                                                       null_mask_col=null_mask)

        df_knn_9 = SkroutzImputation.buildComparisonDf(XX=XX, X_filled=filleds[2],
                                                       disp_name_col=disp_name_col,
                                                       null_mask_col=null_mask)

        df_softimpute = SkroutzImputation.buildComparisonDf(XX=XX, X_filled=filleds[3],
                                                            disp_name_col=disp_name_col,
                                                            null_mask_col=null_mask)

        df_itsvd = SkroutzImputation.buildComparisonDf(XX=XX, X_filled=filleds[4],
                                                       disp_name_col=disp_name_col,
                                                       null_mask_col=null_mask)

        df_mice = SkroutzImputation.buildComparisonDf(XX=XX, X_filled=filleds[5],
                                                      disp_name_col=disp_name_col,
                                                      null_mask_col=null_mask)

        df_nnm = SkroutzImputation.buildComparisonDf(XX=XX, X_filled=filleds[6],
                                                     disp_name_col=disp_name_col,
                                                     null_mask_col=null_mask)

        # X_filled_biscaler = BiScaler()..fit_transform(XX.values)
        # df_biscaler = SkroutzImputation.buildComparisonDf(XX=XX, X_filled=X_filled_biscaler,
        #                                                   disp_name_col=disp_name_col, null_mask_col=null_mask[col])
        # print X_filled_biscaler.shape
        # exit(1)

        # X_filled_matrix_fact = MatrixFactorization().complete(XX.values)
        # df_matrix_fact = SkroutzImputation.buildComparisonDf(XX=XX, X_filled=X_filled_matrix_fact,
        #                                                      disp_name_col=disp_name_col, null_mask_col=null_mask[col])

        comp_df = XX[null_mask].copy()
        dfs = [
            ("knn3", df_knn_3),
            ("knn5", df_knn_5),
            ("knn9", df_knn_9),
            ("soft_impute", df_softimpute),
            ("iter_svd", df_itsvd),
            ("mice", df_mice),
            ("nnm", df_nnm),
            # "matrix_fact": df_matrix_fact,
            # "biscaler": df_biscaler,
        ]

        for count, (cur_key, cur_df) in enumerate(dfs):
            cur_loc = col_id + count + 1
            comp_df.insert(loc=cur_loc, column=col_name + "_" + cur_key, value=cur_df[col_name])

        filepath = "filled/{}.csv".format(col_name)
        comp_df.to_csv(filepath, encoding='utf-8', quoting=csv.QUOTE_ALL)

        return filepath

    def generateComparisonCsvs(self, XX, disp_name_col):
        print "rows:"
        print len(XX)
        print

        print "cols:"
        cols = XX.columns
        print len(cols)
        print

        X_filled_knn3 = KNN(k=3).complete(XX.values)
        X_filled_knn5 = KNN(k=5).complete(XX.values)
        X_filled_knn9 = KNN(k=9).complete(XX.values)
        X_filled_softimpute = SoftImpute().complete(XX.values)
        X_filled_itsvd = IterativeSVD().complete(XX.values)
        X_filled_mice = MICE().complete(XX.values)
        X_filled_nnm = NuclearNormMinimization().complete(XX.values)  # slow

        filleds = [X_filled_knn3,
                   X_filled_knn5, X_filled_knn9, X_filled_softimpute, X_filled_itsvd,
                   X_filled_mice, X_filled_nnm,
                   ]

        filepaths = []
        for col_id in range(len(cols)):
            filepaths.append(
                self.imputing_column(col_id=col_id, XX=XX, disp_name_col=disp_name_col, filleds=filleds)
            )

        return filepaths

    def cpu_cores(self, dataframe, XX_filled):
        df = dataframe.copy()

        col_name = sys._getframe().f_code.co_name  # this_function_name

        null_mask = df.isnull()[col_name]

        nulls_cpu_cores = df[col_name][null_mask]

        filled_df = XX_filled["KNN_5"]

        only_nulls = df[null_mask]

        for (ind, series) in only_nulls.iterrows():
            # (0,0) feature phone, (0,1) for elders, (1,0) smartphone
            isNotSmartphone = series['mobile_type_0'] == 0
            if isNotSmartphone:
                nulls_cpu_cores[ind] = 1
            else:
                nulls_cpu_cores[ind] = int(round(filled_df.loc[ind][col_name]))

        # print nulls_cpu_cores

        df.loc[nulls_cpu_cores.index, col_name] = nulls_cpu_cores

        assert np.NaN not in np.unique(df[col_name])

        return df

    def wireless_charging(self, dataframe, XX_filled):
        df = dataframe.copy()

        col_name = sys._getframe().f_code.co_name  # this_function_name

        null_mask = df.isnull()[col_name]
        nulls = df[col_name][null_mask]

        df.loc[nulls.index, col_name] = 0  # probably does NOT have wireless charging

        assert np.NaN not in np.unique(df[col_name])

        return df

    def removable_battery(self, dataframe, XX_filled):
        df = dataframe.copy()
        this_function_name = sys._getframe().f_code.co_name
        col_name = this_function_name

        null_mask = df.isnull()[col_name]

        index = df[null_mask].index

        filled_df = XX_filled['SoftImpute']

        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name])  # .astype(np.int) #no need for setting type
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0

        return df

    def card_slot(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index

        filled_df = XX_filled["KNN_3"]

        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name])  # .astype(np.int) #no need for setting type
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0

        return df

    def double_back_cam(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index

        filled_df = XX_filled["KNN_9"]

        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name])
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def fast_charge(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index

        filled_df = XX_filled["IterativeSVD"]

        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name])
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def flash(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index

        filled_df = XX_filled["KNN_9"]

        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name])
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def verifyManufacturerIdColsAreFull(self, df):
        manf_cols = df.columns[["manufacturer_id_" in col for col in df.columns]]

        total = 0
        for col in manf_cols:
            total += np.sum(df.isnull()[col])

        assert total == 0

    def protection_dust_resistant(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["KNN_9"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name])
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def protection_rugged(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["KNN_9"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name])
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def protection_water_resistant(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["KNN_9"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name])
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def sensor_accelerometer(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["NuclearNormMinimization"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name])
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def sensor_altimeter(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["NuclearNormMinimization"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name])
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def sensor_barometer(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["NuclearNormMinimization"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name])
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def sensor_compass(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["MICE"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name])
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def sensor_fingerprint(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["MICE"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name])
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def sensor_gyroscope(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["NuclearNormMinimization"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name])
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def sensor_hall(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["MICE"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name])
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def sensor_heartbeat_meter(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["IterativeSVD"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name])
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def sensor_iris_scanner(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["NuclearNormMinimization"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name])
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def sensor_light(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["SoftImpute"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name])
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def sensor_oximeter(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["NuclearNormMinimization"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name])
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def sensor_proximity(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["NuclearNormMinimization"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name])
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def sensor_tango(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["MICE"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name])
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def cam_megapixels(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["KNN_3"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name], decimals=1)
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def cpu_power(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["SoftImpute"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name], decimals=2)
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def ram(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["KNN_5"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name], decimals=2)
        # print df.loc[index, col_name]
        # print np.min(df.loc[index, col_name])
        # print np.max(df.loc[index, col_name])
        # print np.mean(df.loc[index, col_name])
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def release_year(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["MICE"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name], decimals=0)
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def selfie_cam_megapixels(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["KNN_3"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name], decimals=1)
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def speaking_autonomy(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["KNN_5"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name], decimals=1)
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def stanby_autonomy(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["IterativeSVD"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name], decimals=0)
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def storage(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["IterativeSVD"]
        values = np.around(filled_df.loc[index, col_name], decimals=0)
        values[values < 0] = 0  # negative storage is not allowed
        df.loc[index, col_name] = values
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def weight(self, dataframe, XX_filled):
        df = dataframe.copy()
        col_name = sys._getframe().f_code.co_name  # this_function_name
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["KNN_3"]
        values = np.around(filled_df.loc[index, col_name], decimals=1)
        assert len(values[values < 0]) == 0  # negative weights are not allowed
        df.loc[index, col_name] = values
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def connection_network_0(self, dataframe, XX_filled):
        return self.__connection_network(dataframe=dataframe,
                                         col_name=sys._getframe().f_code.co_name,  # this_function_name
                                         XX_filled=XX_filled,
                                         )

    def connection_network_1(self, dataframe, XX_filled):
        return self.__connection_network(dataframe=dataframe,
                                         col_name=sys._getframe().f_code.co_name,  # this_function_name
                                         XX_filled=XX_filled,
                                         )

    def __connection_network(self, dataframe, col_name, XX_filled):
        df = dataframe.copy()
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["KNN_5"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name], decimals=0)
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def operating_system_0(self, dataframe, XX_filled):
        return self.__operating_system(dataframe=dataframe, col_name=sys._getframe().f_code.co_name,
                                       XX_filled=XX_filled)

    def operating_system_1(self, dataframe, XX_filled):
        return self.__operating_system(dataframe=dataframe, col_name=sys._getframe().f_code.co_name,
                                       XX_filled=XX_filled)

    def operating_system_2(self, dataframe, XX_filled):
        return self.__operating_system(dataframe=dataframe, col_name=sys._getframe().f_code.co_name,
                                       XX_filled=XX_filled)

    def operating_system_3(self, dataframe, XX_filled):
        return self.__operating_system(dataframe=dataframe, col_name=sys._getframe().f_code.co_name,
                                       XX_filled=XX_filled)

    def __operating_system(self, dataframe, col_name, XX_filled):
        df = dataframe.copy()

        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["KNN_5"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name], decimals=0)
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def __dimensions(self, dataframe, col_name, XX_filled):
        df = dataframe.copy()

        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["IterativeSVD"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name], decimals=0)
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def height(self, dataframe, XX_filled):
        return self.__dimensions(dataframe=dataframe, col_name=sys._getframe().f_code.co_name, XX_filled=XX_filled)

    def width(self, dataframe, XX_filled):
        return self.__dimensions(dataframe=dataframe, col_name=sys._getframe().f_code.co_name, XX_filled=XX_filled)

    def depth(self, dataframe, XX_filled):
        return self.__dimensions(dataframe=dataframe, col_name=sys._getframe().f_code.co_name, XX_filled=XX_filled)

    def __resolution(self, dataframe, col_name, XX_filled):
        df = dataframe.copy()
        null_mask = df.isnull()[col_name]
        index = df[null_mask].index
        filled_df = XX_filled["MICE"]
        df.loc[index, col_name] = np.around(filled_df.loc[index, col_name], decimals=0)
        # print df.loc[index, col_name]
        assert np.sum(df.isnull()[col_name]) == 0
        return df

    def vertical_resolution(self, dataframe, XX_filled):
        return self.__resolution(dataframe=dataframe, col_name=sys._getframe().f_code.co_name, XX_filled=XX_filled)

    def horizontal_resolution(self, dataframe, XX_filled):
        return self.__resolution(dataframe=dataframe, col_name=sys._getframe().f_code.co_name, XX_filled=XX_filled)

    def createFeatureHowManyEmptyAttributes(self, dataframe, colname='count_empty_fields'):
        df = dataframe.copy()
        XX = df.drop(labels=DROP_COLS, axis=1)
        df.insert(loc=len(df.columns), column=colname, value=-1)

        for ii, (ind, series) in enumerate(XX.iterrows()):
            count_nulls = np.sum(XX.loc[ind].isnull())
            print (ind, count_nulls)
            df.loc[ind, colname] = count_nulls

        assert len(df[colname][df[colname] < 0]) == 0

        return df

    def do(self, csv_in, csv_out):
        df = pd.read_csv(csv_in, index_col=0, encoding='utf-8', quoting=csv.QUOTE_ALL)
        XX = df.drop(labels=DROP_COLS, axis=1)
        nulls_per_attr_series = XX.isnull().sum()

        filledDfDic = {
            "KNN_3": pd.DataFrame(data=KNN(k=3).complete(XX.values), index=XX.index, columns=XX.columns),
            "KNN_5": pd.DataFrame(data=KNN(k=5).complete(XX.values), index=XX.index, columns=XX.columns),
            "KNN_9": pd.DataFrame(data=KNN(k=9).complete(XX.values), index=XX.index, columns=XX.columns),
            "SoftImpute": pd.DataFrame(data=SoftImpute().complete(XX.values), index=XX.index, columns=XX.columns),
            "IterativeSVD": pd.DataFrame(data=IterativeSVD().complete(XX.values), index=XX.index, columns=XX.columns),
            "MICE": pd.DataFrame(data=MICE().complete(XX.values), index=XX.index, columns=XX.columns),
            "NuclearNormMinimization": pd.DataFrame(data=NuclearNormMinimization().complete(XX.values), index=XX.index,
                                                    columns=XX.columns)  # slow
        }

        df = self.createFeatureHowManyEmptyAttributes(
            dataframe=df)  # you MUST execute this line before filling the blanks

        for (ind, nulls) in zip(nulls_per_attr_series.index, nulls_per_attr_series):
            print (ind, nulls)
            if nulls > 0:
                df = getattr(self, ind)(dataframe=df, XX_filled=filledDfDic)

        assert np.sum(df.drop(labels=DROP_COLS, axis=1).isnull().sum()) == 0
        df.drop(labels=FOREVER_DROP_COLS, axis=1).to_csv(csv_out, encoding='utf-8', quoting=csv.QUOTE_ALL)


# TODO verify that all binary values has indeed binary values, and for the rest of the columns as well
# TODO the best way to choose the best method is to compare the distributions graphically (apart from other logical restrictions)
# TODO Perhaps categorical attributes should be treated with the label encoding rather than filling the values separately
# TODO obviously first you have to fill in the blanks and then do feature engineering
if __name__ == "__main__":
    imp = SkroutzImputation()

    #imp.do(csv_in=CSV_FILENAME, csv_out=CSV_OUT_FILEPATH)

    df = pd.read_csv(CSV_FILENAME, index_col=0, encoding='utf-8', quoting=csv.QUOTE_ALL)
    # def crit(xx):
    #     print xx
    #     return np.isnan(xx)
    XX = df.drop(labels=['main_image', 'screen_type', 'display_name', 'future'], axis=1)
    #['main_image', 'screen_type', 'display_name', 'sar']
    #print df['vertical_resolution'][df['vertical_resolution'] != df['vertical_resolution']].count()
    nulls_per_attr_series = XX.isnull().sum()
    print nulls_per_attr_series / len(XX)

    # imp.generateComparisonCsvs(XX=XX, disp_name_col=df['display_name'])
    # df_1 = imp.cpu_cores(dataframe=df)
    # print np.unique(imp.wireless_charging(dataframe=df)['wireless_charging'])
    # print np.unique(imp.removable_battery(dataframe=df)['removable_battery'])
    # print np.unique(imp.card_slot(dataframe=df)['card_slot'])
    # print np.unique(imp.double_back_cam(dataframe=df)['double_back_cam'])
    # print np.unique(imp.fast_charge(dataframe=df)['fast_charge'])
    # print np.unique(imp.flash(dataframe=df)['flash'])
    # print np.sum(df['future'])
    # imp.verifyManufacturerIdColsAreFull(df=df)
    # print np.unique(imp.protection_dust_resistant(dataframe=df)['protection_dust_resistant'])
    # print np.unique(imp.protection_rugged(dataframe=df)['protection_rugged'])
    # print np.unique(imp.sensor_accelerometer(dataframe=df)['sensor_accelerometer'])
    # print np.unique(imp.sensor_altimeter(dataframe=df)['sensor_altimeter'])
    # print np.unique(imp.sensor_barometer(dataframe=df)['sensor_barometer'])
    # print np.unique(imp.sensor_compass(dataframe=df)['sensor_compass'])
    # print np.unique(imp.sensor_fingerprint(dataframe=df)['sensor_fingerprint'])
    # print np.unique(imp.sensor_gyroscope(dataframe=df)['sensor_gyroscope'])
    # print np.unique(imp.sensor_hall(dataframe=df)['sensor_hall'])
    # print np.unique(imp.sensor_heartbeat_meter(dataframe=df)['sensor_heartbeat_meter'])
    # print np.unique(imp.sensor_iris_scanner(dataframe=df)['sensor_iris_scanner'])
    # print np.unique(imp.sensor_light(dataframe=df)['sensor_light'])
    # print np.unique(imp.sensor_oximeter(dataframe=df)['sensor_oximeter'])
    # print np.unique(imp.sensor_proximity(dataframe=df)['sensor_proximity'])
    # print np.unique(imp.sensor_tango(dataframe=df)['sensor_tango'])
    # print np.unique(imp.cam_megapixels(dataframe=df)['cam_megapixels'])
    # print np.unique(imp.cpu_power(dataframe=df)['cpu_power'])
    # print np.unique(imp.ram(dataframe=df)['ram'])
    # print np.unique(imp.release_year(dataframe=df)['release_year'])
    # print np.unique(imp.selfie_cam_megapixels(dataframe=df)['selfie_cam_megapixels'])
    # print np.unique(imp.speaking_autonomy(dataframe=df)['speaking_autonomy'])
    # print np.unique(imp.stanby_autonomy(dataframe=df)['stanby_autonomy'])
    # print np.unique(imp.storage(dataframe=df)['storage'])
    # print np.unique(imp.weight(dataframe=df)['weight'])
    # print np.unique(imp.connection_network_0(dataframe=df)['connection_network_0'])
    # print np.unique(imp.connection_network_1(dataframe=df)['connection_network_1'])
    # print np.unique(imp.operating_system_0(dataframe=df)['operating_system_0'])
    # print np.unique(imp.operating_system_1(dataframe=df)['operating_system_1'])
    # print np.unique(imp.operating_system_2(dataframe=df)['operating_system_2'])
    # print np.unique(imp.operating_system_3(dataframe=df)['operating_system_3'])
    # print np.unique(imp.height(dataframe=df)['height'])
    # print np.unique(imp.width(dataframe=df)['width'])
    # print np.unique(imp.depth(dataframe=df)['depth'])
    # print np.unique(imp.vertical_resolution(dataframe=df)['vertical_resolution'])
    # print np.unique(imp.horizontal_resolution(dataframe=df)['horizontal_resolution'])

    print "DONE"
    #os.system('play --no-show-progress --null --channels 1 synth {} sine {}'.format(0.5, 800))
