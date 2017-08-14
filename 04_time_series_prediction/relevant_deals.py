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


def printgr(obj):
    print repr(obj).decode('unicode-escape')


class RelevantDeals(object):
    def __init__(self, df_all=None, deal_df=None):
        super(RelevantDeals, self).__init__()
        if df_all is None:
            df_all = pd.concat((pd.read_csv('../mobiles_03_train.csv', index_col=0, encoding='utf-8',
                                            quoting=csv.QUOTE_ALL),
                                pd.read_csv('../mobiles_03_test.csv', index_col=0, encoding='utf-8',
                                            quoting=csv.QUOTE_ALL)), axis=0)

        if deal_df is None:
            deal_df = pd.read_csv('../mobiles_04_deals_display.csv', index_col=0, encoding='utf-8',
                                  quoting=csv.QUOTE_ALL)

        assert set(df_all.index) == set(deal_df.index), "they must have the same index"

        self.df_all = df_all
        self.deal_df = deal_df

        self.nn = NearestNeighbors(n_neighbors=len(df_all),  # for simple nearest neighbors,not play a role
                                   p=2,  # minkowski distance
                                   # radius=1., leaf_size=30,  #for simple nearest neighbors it does not play a role
                                   )

        self.nn.fit(df_all)

    def getSome(self, target_ind):
        """returns a list with SKU ids. Keep the first ones to get the most relevant
        target_ind is pandas single index value"""
        if target_ind in self.df_all.index:
            curX = self.df_all.loc[target_ind].values[np.newaxis]
            neighbors_sorted = self.nn.kneighbors(X=curX,  # we drop the first because it is itself
                                                  n_neighbors=len(self.df_all), return_distance=False).flatten()[1:]
            neighbors_sorted_inds = [self.df_all.index[neighbor] for neighbor in neighbors_sorted]

            cur_deal_metric = self.deal_df.loc[target_ind]['deal_metric']
            better_deals = self.deal_df[self.deal_df['deal_metric'] > cur_deal_metric]
            better_deal_inds = better_deals.index

            # filtering out only better deals without losing neighbor order
            relevant_better_deals = np.array([neighbor_ind for neighbor_ind in neighbors_sorted_inds
                                              if neighbor_ind in better_deal_inds])

            return relevant_better_deals
        else:
            return None


if __name__ == "__main__":
    df_deal = pd.read_csv('../mobiles_04_deals_display.csv', index_col=0, encoding='utf-8',
                          quoting=csv.QUOTE_ALL)

    df = pd.concat((pd.read_csv('../mobiles_03_train.csv', index_col=0, encoding='utf-8',
                                quoting=csv.QUOTE_ALL),
                    pd.read_csv('../mobiles_03_test.csv', index_col=0, encoding='utf-8',
                                quoting=csv.QUOTE_ALL)), axis=0)

    rd = RelevantDeals(deal_df=df_deal, df_all=df)

    #targetIndex = df.index[347]  # Apple iphone 6 (16GB)
    leeco = df_deal[["LeEco Le Max 2 (128GB)" in cur_name for cur_name in df_deal['display_name']]]
    targetIndex = leeco.index[0]
    print df_deal.loc[targetIndex]

    print rd.getSome(target_ind=targetIndex)[:10]
