from collections import OrderedDict

import numpy as np

from helpers.py_helper import convert_date_str_to_info
from pandas import read_csv, DataFrame
import csv

from price_history import PriceHistory
from skroutz_mobile import SkroutzMobile


class MobAttrsPriceHistoryMerger(object):
    DATE_INFO_LEN = 6
    PRICE_IND = -1  # last column is the price

    def __init__(self, mobs_norm_path, price_history_csv):
        super(MobAttrsPriceHistoryMerger, self).__init__()

        self.df = read_csv(mobs_norm_path, index_col=0, encoding='utf-8', quoting=csv.QUOTE_ALL).drop(
            labels=SkroutzMobile.PRICE_COLS
            # we should remove information that is directly related with price since this was only info about min and maximum visible price
            # However this could help the model if we are really predicting future dates, which we should remove from training
            # For now we will follow the safest path (of not-cheating fooling ourselves) and we will remove those attributes,
            # however in the future after we successfully remove future dates and put them in a test set it worths to see
            # if these two attributes help our model. BECAREFUL that these attributes should have been collected BEFORE
            # we have the future price history. In other words we want the price min and price max to refer to past dates
            , axis=1)

        ph = PriceHistory(price_history_csv)
        self.seqs = ph.extractAllSequences()

    def get_mob_attrs_indices(self):
        # - len(SkroutzMobile.PRICE_COLS) no need to do that since we have already dropped the columns
        return np.arange(0, self.df.shape[1])

    def get_date_indices(self):
        mob_attrs_inds = self.get_mob_attrs_indices()
        return np.arange(mob_attrs_inds[-1] + 1, mob_attrs_inds[-1] + 1 + self.DATE_INFO_LEN)

    def get_table(self, seqs=None, df=None, normalize_dates=False, normalize_price=False):
        mylist = list(self.generate_rows(seqs=self.seqs if seqs is None else seqs,
                                         df=self.df if df is None else df))

        data = np.array([item[1] for item in mylist])

        data_norm_dates = self.normalize_date_columns(arr=data) if normalize_dates else data
        data_processed = self.normalize_price_col(arr=data_norm_dates) if normalize_price else data_norm_dates

        return DataFrame(index=[item[0] for item in mylist], data=data_processed)

    def generate_rows(self, seqs, df):
        for seq in seqs:
            if seq.name in df.index:
                for row in self.generate_rows_per_seq(seq=seq, df=df):
                    yield row
            else:
                pass

    def generate_rows_per_seq(self, seq, df):
        assert seq.name in df.index

        sku_id = seq.name

        cur_mob = df.loc[sku_id]
        cur_mob_len = len(cur_mob)

        for cur_date_key, cur_price in seq.iteritems():
            cur_date_info = np.array(convert_date_str_to_info(cur_date_key).values()).astype(np.float32)

            data_row = np.hstack((cur_mob, cur_date_info, cur_price))

            assert len(data_row) == len(cur_date_info) + 1 + cur_mob_len
            assert np.all(np.logical_not(np.isnan(data_row)))

            yield sku_id, data_row

    def get_normalize_price_metrics(self, arr):
        price_col = arr[:, self.PRICE_IND]
        price_mean = np.mean(price_col)
        price_std = np.std(price_col)

        return {-1: (price_mean, price_std)}

    def get_normalize_date_metrics(self, arr):
        dic = OrderedDict()
        for cur_date_ind in self.get_date_indices():
            cur_col = arr[:, cur_date_ind]
            cur_mean = np.mean(cur_col)
            cur_std = np.std(cur_col)
            dic[cur_date_ind] = cur_mean, cur_std

        return dic

    def normalize_price_col(self, arr):
        arr_norm = arr.copy()

        price_col = arr_norm[:, self.PRICE_IND]
        price_mean = np.mean(price_col)
        price_std = np.std(price_col)
        arr_norm[:, self.PRICE_IND] = (price_col - price_mean) / price_std

        return arr_norm

    def normalize_date_columns(self, arr):
        arr_norm = arr.copy()

        for cur_date_ind in self.get_date_indices():
            cur_col = arr_norm[:, cur_date_ind]
            cur_mean = np.mean(cur_col)
            cur_std = np.std(cur_col)
            arr_norm[:, cur_date_ind] = (cur_col - cur_mean) / cur_std

        return arr_norm

    @staticmethod
    def normalize_df(df, dic_metrics, verbose):
        dataframe = df.copy()
        if verbose:
            print dataframe.shape
        for key, val in dic_metrics.iteritems():
            dataframe.iloc[:, key] = (dataframe.iloc[:, key] - val[0]) / val[1]

        return dataframe
