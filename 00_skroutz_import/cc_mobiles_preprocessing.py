# -*- coding: UTF-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import sys
from advanced_one_hot_encoder import AdvancedOneHotEncoder
import math
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import re
import os
import csv

def printgr(obj):
    print repr(obj).decode('unicode-escape')


# this = sys.modules[__name__]
# print this

class MobileBasicPreprocessing(object):
    def __init__(self, filename):
        super(MobileBasicPreprocessing, self).__init__()
        self.df = pd.read_csv('{}.csv'.format(filename), index_col=0, encoding='utf-8', quoting=csv.QUOTE_ALL)

    # TODO Have the decision tree to create extra features for the neural network

    real_cols = [
        'battery_capacity',
        'cam_megapixels',
        'cpu_power',
        'diagonal_size',
        'price_max',
        'price_min',
        'ram',
        'release_year',
        'reviews_count',
        'reviewscore',
        'sar',
        'selfie_cam_megapixels',
        'shop_count',
        'speaking_autonomy',
        'stanby_autonomy',
        'storage',
        'weight',
    ]

    binary_cols = [
        'wireless_charging',
        'removable_battery',
        'card_slot',
        'double_back_cam',
        'fast_charge',
        'flash',
        'future',
        'connectivity_bluetooth',
        'connectivity_lightning',
        'connectivity_minijack',
        'connectivity_nfc',
        'connectivity_usb',
        'connectivity_usb_type_c',
        'connectivity_wifi',
        'control_natural_keyboard',
        'control_touchscreen',
        'protection_dust_resistant',
        'protection_rugged',
        'protection_water_resistant',
        'sensor_accelerometer',
        'sensor_altimeter',
        'sensor_barometer',
        'sensor_compass',
        'sensor_fingerprint',
        'sensor_gyroscope',
        'sensor_hall',
        'sensor_heartbeat_meter',
        'sensor_iris_scanner',
        'sensor_light',
        'sensor_oximeter',
        'sensor_proximity',
        'sensor_tango',
    ]

    categorical_cols = [
        'connection_network',
        'mobile_type',  #(0,0) feature phone, (0,1) for elders, (1,0) smartphone
        'manufacturer_id',
        'operating_system',
    ]

    other_cols = [
        'display_name',  # don't touch this
        'dimensions',  # w, h, d, w x h, h x d, w x d, w x h x d
        'screen_resolution',  # height, width, height x width
        'sim',

        'cpu_cores',

        'screen_type',
        'main_image',
    ]

    drop_cols = [
        'id',
        'name',
        'plain_spec_summary',
        'pn',
        'virtual',  # they are all zero
    ]

    def printUniquesForCategoricalAttrs(self):
        for col in self.categorical_cols:
            uniques = np.unique(self.df[col])
            print "{}({})".format(col, len(uniques))
            printgr(uniques)
            print

    def dropUnnecessaryCols(self):
        orig_col_len = len(self.df.columns)
        self.df.drop(labels=self.drop_cols, axis=1, inplace=True)
        assert orig_col_len == len(self.df.columns) + len(self.drop_cols)
        return self.df

    def setRealAndBinaryColsLast(self):
        self.df = self.df[self.getRealAndBinaryColsLast()]
        return self.df

    def getRealAndBinaryColsLast(self):
        cols = self.df.columns
        rest_cols = list(set(cols).difference(self.real_cols + self.binary_cols))
        return rest_cols + self.binary_cols + self.real_cols

    def getAllCols(self):
        return set(self.real_cols).union(
            self.binary_cols).union(
            self.categorical_cols).union(
            self.other_cols).union(self.drop_cols)

    def verifyAllColumnsAreIncluded(self):
        verification = set(self.df.columns) == self.getAllCols()
        assert verification
        return verification
        # print set(self.df.columns).difference(self.getAllCols())

    DIMS_SEPS = [u'x', u'X', u'×', u'χ', u'Χ']  # english, greek and special symbols
    SCREENRES_SEP = [u'x']

    def process_screen_resolution(self, df):
        col_name = 'screen_resolution'
        col = df[col_name]

        def strs_to_ints(strs):
            stripped_strs = [thestr.strip() for thestr in strs]
            try:
                return [int(thestr) for thestr in stripped_strs]
            except ValueError:
                integers = []
                for thestr in stripped_strs:
                    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", thestr)
                    integers.append(int(nums[0]))  # just get the first number
                return integers

        def splitter(theinput):
            """vertical is the large resolution, horizontal is the small one
            this is typical because all mobile phones are made to be portrait (not landscape)"""
            if isinstance(theinput, basestring):
                splits = [theinput.split(sep) for sep in self.DIMS_SEPS]
                lengths = np.array([len(splitted) for splitted in splits])

                pos_arr = np.argwhere(lengths == 2).flatten()
                assert 0 <= len(pos_arr) <= 1, "we are expecting to have only one separator"
                if len(pos_arr) == 1:
                    strs = splits[pos_arr[0]]
                    resolution = np.sort(strs_to_ints(strs))[::-1]
                    return resolution[0], resolution[1]
                else:
                    return theinput, theinput
            else:
                return theinput, theinput

        tuple_list = [splitter(theinput=cur_in) for cur_in in col]

        vertical_res = np.array([tpl[0] for tpl in tuple_list])[np.newaxis].T
        horizontal_res = np.array([tpl[1] for tpl in tuple_list])[np.newaxis].T

        arr = np.hstack((vertical_res, horizontal_res))

        dataframe = pd.DataFrame(data=arr, index=col.index, columns=[
            'vertical_resolution', 'horizontal_resolution',
        ])

        return pd.concat((df.drop(labels=[col_name], axis=1), dataframe), axis=1)

    def process_dimensions(self, df):
        col_name = 'dimensions'
        col = df[col_name]

        def dim_strs_to_floats(dims_strs):
            stripped_dim_strs = [dim_str.strip() for dim_str in dims_strs]
            try:
                return [float(dim_str) for dim_str in stripped_dim_strs]
            except ValueError:
                dims = []
                for dim_str in stripped_dim_strs:
                    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", dim_str)
                    dims.append(float(nums[0]))  # just get the first number
                return dims

        def splitter(theinput):
            empty = np.NaN, np.NaN, np.NaN

            if isinstance(theinput, basestring):
                dic = dict([(sep, theinput.split(sep)) for sep in self.DIMS_SEPS])
                lengths = np.array([len(splitted) for splitted in dic.values()])

                four_pos_arr = np.argwhere(lengths == 4).flatten()

                three_pos_arr = np.argwhere(lengths == 3).flatten()
                assert 0 <= len(three_pos_arr) <= 1, "we are expecting to have only one separator"
                if len(three_pos_arr) == 1:
                    dims_strs = dic.values()[three_pos_arr[0]]
                    dims = np.sort(dim_strs_to_floats(dims_strs))[::-1]
                    return dims[0], dims[1], dims[2]
                elif len(four_pos_arr) == 1:
                    dims_strs = dic.values()[four_pos_arr[0]]
                    dims = np.sort(dim_strs_to_floats(dims_strs))[::-1][:3]  # keep only the three largest
                    # print dims
                    return dims[0], dims[1], dims[2]
                else:
                    # print "invalid input:"
                    # print theinput
                    #return theinput, theinput, theinput
                    return empty
            else:
                # print "invalid input:"
                # print theinput
                #return theinput, theinput, theinput
                return empty

        tuple_list = [splitter(theinput=cur_in) for cur_in in col]

        # print np.array([[tpl[0] for tpl in tuple_list]]).T
        heights = np.array([tpl[0] for tpl in tuple_list])[np.newaxis].T
        widths = np.array([tpl[1] for tpl in tuple_list])[np.newaxis].T
        depths = np.array([tpl[2] for tpl in tuple_list])[np.newaxis].T
        # print "heights"
        # print heights.dtype
        # print heights.shape
        # print "widths"
        # print widths.dtype
        # print widths.shape

        arr = np.hstack((heights, widths, depths))

        # print arr
        dataframe = pd.DataFrame(data=arr, index=col.index, columns=[
            'height', 'width', 'depth',
        ])
        # table = np.apply_along_axis(func1d=splitter, axis=0, arr=arr.values)

        return pd.concat((df.drop(labels=[col_name], axis=1), dataframe), axis=1)

    @staticmethod
    def checkSeparatorsUsedInScreenResolutionFeature(df):
        seps = set()
        for res in df['screen_resolution']:
            cur_sep = None
            for sep in instance.DIMS_SEPS:
                try:
                    if sep in res:
                        cur_sep = sep
                        break
                except TypeError:
                    pass
            if cur_sep is None:
                print "invalid value {}".format(res)
            else:
                seps.add(cur_sep)
        return seps

    @staticmethod
    def process_sim(df):
        df_new = df.copy()
        df_new['sim'] = [1 if vv == "Single" else 2 for vv in df['sim']]
        return df_new

    # TODO deal better with ordinal values https://github.com/fabianp/mord
    CPU_CORES_ASSIGNMENT = {
        '1': 1,
        '2': 2,
        '2+2': 3,
        '3': 4,
        '4': 5,
        '4+2': 6,
        '4+4': 7,
        '4+4+2': 8,
        '5': 9,
        '7': 10,
        '8': 11,
    }

    def process_cpu_cores(self, df):
        df_new = df.copy()
        df_new['cpu_cores'] = [self.CPU_CORES_ASSIGNMENT[vv] if isinstance(vv, basestring) else vv
                               for vv in df['cpu_cores']]
        return df_new


if __name__ == "__main__":
    instance = MobileBasicPreprocessing(filename='non_processed_mobiles')
    print instance.verifyAllColumnsAreIncluded()

    df_0 = instance.dropUnnecessaryCols()
    print len(df_0.columns)

    # print getattr(obj, 'binary_cols')
    df_1 = instance.setRealAndBinaryColsLast()

    instance.printUniquesForCategoricalAttrs()
    # print df
    # df = AdvancedOneHotEncoder().encodePandasColAndMerge(data_frame=df, col_name='connection_network')
    # print df

    df_2 = AdvancedOneHotEncoder().encodePandasColAndMerge(data_frame=df_1, col_name='connection_network',
                                                           check_to_null=lambda vv:
                                                           not isinstance(vv, basestring) and math.isnan(vv))
    print len(df_2.columns)

    df_3 = AdvancedOneHotEncoder().encodePandasColAndMerge(data_frame=df_2, col_name='mobile_type')
    print len(df_3.columns)

    df_4 = AdvancedOneHotEncoder().encodePandasColAndMerge(data_frame=df_3, col_name='manufacturer_id',
                                                           check_to_null=lambda vv:
                                                           not isinstance(vv, basestring) and math.isnan(vv))
    print len(df_4.columns)

    df_5 = AdvancedOneHotEncoder().encodePandasColAndMerge(data_frame=df_4, col_name='operating_system',
                                                           check_to_null=lambda vv:
                                                           not isinstance(vv, basestring) and math.isnan(vv))
    print len(df_5.columns)

    df_6 = instance.process_dimensions(df_5)
    print len(df_6.columns)

    df_7 = instance.process_screen_resolution(df_6)
    print len(df_7.columns)

    # print np.unique(df_7['sim'])
    df_8 = instance.process_sim(df=df_7)
    print len(df_8.columns)

    # print "screen_type:"
    # print np.unique(df_8['screen_type'])
    # print len(np.unique(df_8['screen_type']))

    df_9 = instance.process_cpu_cores(df=df_8)
    # print df_9['cpu_cores']
    print len(df_9.columns)

    df_9.to_csv("../mobiles_basic_preprocess.csv", encoding='utf-8', quoting = csv.QUOTE_ALL)

    os.system('play --no-show-progress --null --channels 1 synth {} sine {}'.format(0.5, 800))
