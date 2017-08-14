# -*- coding: UTF-8 -*-
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
from math import floor, ceil

class MySplitTrainTest(object):
    def __getNumberOfBins(self, approx_factor, yy):
        return int(round(len(yy) / approx_factor))

    def __getBins(self, number_of_bins, yy):
        return pd.qcut(yy.values, number_of_bins, retbins=False, labels=None, duplicates='drop')

    def __splitable(self, discr, test_size):
        """make sure test_size is a fraction from 0 to 1"""
        test_len = floor(len(discr) * test_size)  #floor is better than round here
        classes = discr.groupby(discr).agg(['count'])
        #print len(classes)
        minimum_count = np.min(classes).values[0]
        return minimum_count > 1 and test_len >= len(classes)

    def __getDiscr(self, bins, yy):
        return pd.Series([int(round(np.mean((interval.left, interval.right)))) for interval in bins], index=yy.index)

    def splitTrainTestForRegression(self, XX_dataframe, yy_series, test_size=0.1, random_state=None, debug=False):
        """returns tuple of train indices and test indices
        recall that the indices are NOT the index of the dataframe but rather the actual location"""

        sss = StratifiedShuffleSplit(n_splits=2, test_size=test_size, random_state=random_state)  # 10% for testing

        #discr = np.round(yy_series).astype(np.int)  # first line of offence
        discr = yy_series.round()

        if self.__splitable(discr=discr, test_size=test_size):
            if debug:
                print "rounding worked"
            return sss.split(XX_dataframe, discr).next()
        else:
            cur_approx_factor = 2
            repeat = True
            while repeat:
                number_of_bins = self.__getNumberOfBins(approx_factor=cur_approx_factor, yy=yy_series)
                if number_of_bins <= 1:
                    repeat = False

                bins = self.__getBins(number_of_bins=number_of_bins, yy=yy_series)
                discr = self.__getDiscr(bins=bins, yy=yy_series)
                if self.__splitable(discr=discr, test_size=test_size):
                    if debug:
                        print "approx factor is {}".format(cur_approx_factor)
                    return sss.split(XX_dataframe, discr).next()

                cur_approx_factor += 1
            #pd.concat((yy, discr), axis=1).sample(10, random_state=random_state)
            raise Exception("we cannot split it because the dataset has only one item?.. you are in an extreme case")
