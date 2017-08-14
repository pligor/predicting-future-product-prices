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
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
from relevant_deals import RelevantDeals
from sklearn.model_selection import StratifiedShuffleSplit


def printgr(obj):
    print repr(obj).decode('unicode-escape')


class DealTargetsGenerator(object):
    """
    - read csv to have combinations along with targets
    - For every batch of combinations have a way to create an input with product As and product Bs and also the corresponding targets.
    So this will be something like A+B ~= 280 features and if batch is 50 then we have 50x280
    """

    def __init__(self, df_deal):
        super(DealTargetsGenerator, self).__init__()
        self.df_deal = df_deal

    def generate(self, full_index, nn=20):
        """nn is how many
        df_deal = pd.read_csv('../mobiles_04_deals_display.csv', index_col=0, encoding='utf-8',
                          quoting=csv.QUOTE_ALL)
        instance = DealTargetsGenerator(df_deal=df_deal)
        df_targets = instance.generate(full_index=df_deal.index)
        df_targets.to_csv('../mobiles_05_comparison_targets.csv', encoding='utf-8', quoting=csv.QUOTE_ALL)
        """
        # TODO run an assertion to verify that the generated targets are unique
        rd = RelevantDeals()

        tuples = np.empty((0, 3))
        for ind in full_index:
            # print ind
            neighbor_inds = rd.getSome(target_ind=ind)[:nn]
            # print len(neighbor_inds)
            for neighbor_ind in neighbor_inds:
                tworows = np.array([[ind, neighbor_ind, 1],
                                    [neighbor_ind, ind, 0]])

                # print tuples.shape

                tuples = np.vstack((tuples, tworows))
                # print tuples.shape

        print len(tuples)
        return pd.DataFrame(data=tuples, columns=['worst_deal', 'better_deal', 'binary_target'])

    @staticmethod
    def splitTrainTest(save=False):
        # sss = StratifiedShuffleSplit()
        df_comp = pd.read_csv('../mobiles_05_comparison_targets.csv', index_col=0, encoding='utf-8',
                              quoting=csv.QUOTE_ALL)

        df_test = pd.read_csv('../mobiles_03_test.csv', index_col=0, encoding='utf-8',
                              quoting=csv.QUOTE_ALL)

        test_index = df_test.index

        rows_that_contain_test_indices = [ind for ind, product_ind in df_comp.iterrows()
                                          if product_ind['worst_deal'] in test_index or
                                          product_ind['better_deal'] in test_index]

        df_comp_test = df_comp.loc[rows_that_contain_test_indices]
        df_comp_train = df_comp.loc[[ind for ind in df_comp.index if ind not in rows_that_contain_test_indices]]

        assert set(df_comp_test.index).union(df_comp_train.index) == set(df_comp.index)

        assert len(df_comp_test[df_comp_test['binary_target'] == 1]) == len(
            df_comp_test[df_comp_test['binary_target'] == 0])

        assert len(df_comp_train[df_comp_train['binary_target'] == 0]) == len(
            df_comp_train[df_comp_train['binary_target'] == 1])

        if save:
            df_comp_train.to_csv('../mobiles_05_comp_train.csv', encoding='utf-8', quoting=csv.QUOTE_ALL)
            df_comp_test.to_csv('../mobiles_05_comp_test.csv', encoding='utf-8', quoting=csv.QUOTE_ALL)

        return df_comp_train, df_comp_test


if __name__ == "__main__":
    # df = pd.concat((,
    #                 ), axis=0)
    # df_train = pd.read_csv('../mobiles_03_train.csv', index_col=0, encoding='utf-8',
    #                        quoting=csv.QUOTE_ALL)
    # df_train = pd.read_csv('../mobiles_03_train.csv', index_col=0, encoding='utf-8',
    #                        quoting=csv.QUOTE_ALL)

    DealTargetsGenerator.splitTrainTest(save=True)

    # print [len(product_ind) for product_ind in df_comp.iterrows()]

    # print len(df)
