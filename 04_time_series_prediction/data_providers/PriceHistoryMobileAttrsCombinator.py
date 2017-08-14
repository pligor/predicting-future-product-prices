import numpy as np
import pandas as pd
import csv


class PriceHistoryMobileAttrsCombinator(object):
    def __init__(self, mobiles_path='../data/mobiles/mobiles_norm.csv'):
        super(PriceHistoryMobileAttrsCombinator, self).__init__()
        self.mobiles_path = mobiles_path
        self.df = pd.read_csv(mobiles_path, index_col=0, encoding='utf-8', quoting=csv.QUOTE_ALL)

    def combine(self, npz_in):
        """combine price history with mobile attributes for each instance"""

        dic = np.load(npz_in)
        sku_ids = dic['sku_ids']
        df = self.df
        # mobiles_arr = np.empty(shape=(0, df.shape[1]))
        # mobiles_arr.shape
        count_key_errors = 0
        key_errors = set()

        keys = dic.keys()[:]  # 'inputs', 'sku_ids', 'sequence_masks', 'targets', 'sequence_lengths'
        # keys.append('mobile_attrs')
        mobiles_list = []
        # list_dic = dict([(key, []) for key in keys])
        inds = []
        for ii, sku_id in enumerate(sku_ids):
            try:
                # mobiles_arr[ii] = df.loc[sku_id].values
                mobiles_list.append(df.loc[sku_id].values)
                # for key in keys:
                #     list_dic[key].append(dic[key][ii])
                inds.append(ii)  # because not all indices make it, because of key error we lose some
            except KeyError:
                key_errors.add(sku_id)
                count_key_errors += 1

        final_dic = {
            'mobile_attrs': np.array(mobiles_list)
        }

        for key in keys:
            final_dic[key] = dic[key][inds]

        # new_key = 'mobile_attrs'
        # list_dic[new_key] = mobiles_list
        # final_dic = dict([(key, np.array(val)) for key, val in list_dic.iteritems()])
        # length = len(final_dic.values()[0])
        # assert np.all([len(val) == length for key, val in list_dic.iteritems()]), 'all must be of equal length'
        # return final_dic, count_key_errors, key_errors
        #mobiles_arr = np.array(mobiles_list)

        return final_dic, inds, count_key_errors, key_errors

    # def filter_key(self, npz_in, inds, datakey):
    #     """This function has some serious memory issue that is why we need to process things more carefully:
    #     1) First attempt to simply process each key separately, and first we process only the mobiles this should reduce the memory footprint
    #     inputs', 'sku_ids', 'sequence_masks', 'targets', 'sequence_lengths"""
    #
    #     dic = np.load(npz_in)
    #     df = self.df
    #
    #     keys = dic.keys()[:]  # 'inputs', 'sku_ids', 'sequence_masks', 'targets', 'sequence_lengths'
    #     assert datakey in dic.keys(), 'provided data key must of course exist in the dictionary of the npz'
    #
    #     thelist = []
    #     # list_dic = dict([(key, []) for key in keys])
    #
    #     inds = []
    #     for ii in inds:
    #         # mobiles_arr[ii] = df.loc[sku_id].values
    #         thelist.append(df.loc[sku_id].values)
    #
    #         # for key in keys:
    #         #     list_dic[key].append(dic[key][ii])
    #
    #     # new_key = 'mobile_attrs'
    #     # list_dic[new_key] = mobiles_list
    #     # final_dic = dict([(key, np.array(val)) for key, val in list_dic.iteritems()])
    #     # length = len(final_dic.values()[0])
    #     # assert np.all([len(val) == length for key, val in list_dic.iteritems()]), 'all must be of equal length'
    #     # return final_dic, count_key_errors, key_errors
    #     mobiles_arr = np.array(thelist)
    #
    #     return mobiles_arr, inds



