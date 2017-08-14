# -*- coding: UTF-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import csv

from price_history_trait import PriceHistoryTrait


class PriceHistory(PriceHistoryTrait):
    # CSV_FILEPATH = "../price_history_02_with_seq_start.csv"

    SPECIAL_COLS = ['sequence_length', 'seq_start']

    PRICE_COLS = ['price_min', 'price_max']
    TARGET_COL = 'price_min'

    def __init__(self, csv_filepath):
        super(PriceHistory, self).__init__()

        self.orig_df = pd.read_csv(csv_filepath, index_col=0, encoding='utf-8',
                                   quoting=csv.QUOTE_ALL)

        self.df = self.orig_df.drop(labels=self.SPECIAL_COLS, axis=1)

    # def extractSequenceByLocation(self, iloc):
    #     return super(PriceHistory, self).extractSequenceByLocation(self.orig_df, iloc)

    def extractAllSequences(self):
        return super(PriceHistory, self).extractAllSequences(self.orig_df)


if __name__ == "__main__":
    print "DONE"
