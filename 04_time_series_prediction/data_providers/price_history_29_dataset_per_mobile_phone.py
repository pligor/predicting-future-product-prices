from __future__ import division
import csv
import numpy as np
import pandas as pd

from data_providers.price_history_27_pack import PriceHistory27Pack
from mylibs.py_helper import convert_date_str_to_info
from price_history_trait import PriceHistoryTrait


class PriceHistoryDatasetPerMobilePhone(PriceHistoryTrait):
    """
    for splitting dataset to training and testing set there are different ways to do it:
    - We could use as dataset phones we have never seen before
        but this would make our problem a little bit more difficult for our Neural Network
    - Second way is to be stratified. Meaning that we are going to test choosing from all cell phones

    The goal is to choose the golden edge between something that it will not be too hard for our model (since the problem
    of price prediction is already hard by itself) and also that makes sense and gives us the truth about unseen cases.
    So we need to pick a window in time randomly for every product and have this as our testing dataset.

    Another idea is to have products that are close to each other (define distance) and have one
    in training set and the other in testing set. But this does not make full sense since in general we are going to have
    most of the products available. I guess both cases should be tested because we will eventually have new products and
    we will need to know how our algorithm is going to perform on them if only a single window can be generated from the dataset
    """

    def __init__(self, random_state):
        super(PriceHistoryDatasetPerMobilePhone, self).__init__()
        self.random_state = np.random if random_state is None else random_state

    MIN_TARGET_SEQ_LEN = 1
    MIN_INPUT_SEQ_LEN = 2 * MIN_TARGET_SEQ_LEN

    PRICE_HISTORY_SPECIAL_COLS = 2

    def genSaveDictionary(self, csv_in, window_len, npz_out=None):
        df = pd.read_csv(csv_in, index_col=0, quoting=csv.QUOTE_ALL, encoding='utf-8')

        dic = {}
        for sku_id, data_dic in self.__processDataFrame(df, window_len):
            dic[str(sku_id)] = data_dic

        if npz_out is not None:
            np.savez(npz_out, **dic)

        return dic

    def __processDataFrame(self, df, window_len):
        for sku_id, row in df.iterrows():
            row_norm = self.instance_scale_normalization(row)

            seq = self.extractSequence(row_norm)

            if len(seq) >= window_len * 2:
                test_set = seq[-window_len:]

                train_part = seq[:-window_len]

                train_subseqs = list(self.getSeqWindows(sequence=train_part, window_length=window_len))
                train_set = np.array(train_subseqs)
                train_dates = np.array([subseq.index for subseq in train_subseqs])

                yield sku_id, {
                    "train": train_set,
                    "train_dates": train_dates,
                    "test": test_set,
                }
            else:
                continue

    @staticmethod
    def getSeqWindows(sequence, window_length):
        seq_length = len(sequence)
        assert (window_length <= seq_length)
        indices = np.arange(seq_length - window_length + 1)
        # windows = (indices + np.arange(0, window_length)[np.newaxis].T).T
        windows = (indices + np.arange(0, window_length)[:, None]).T
        for window in windows:
            cur_slice = slice(window[0], window[-1] + 1)
            yield sequence[cur_slice]

    def instance_scale_normalization(self, item):
        """normalize the scale of a row of the dataframe"""
        seq = self.extractSequence(row=item)
        std = np.std(seq)
        if std == 0:
            return item
        else:
            newitem = item.copy()
            cur_slice = slice(0, -self.PRICE_HISTORY_SPECIAL_COLS)
            newitem.iloc[cur_slice] = item.iloc[cur_slice] / std

            return newitem
