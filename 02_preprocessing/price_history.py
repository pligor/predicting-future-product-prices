# -*- coding: UTF-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import csv

from os.path import isfile

from price_history_trait import PriceHistoryTrait


class PriceHistory(PriceHistoryTrait):
    SPECIAL_COLS = ['sequence_length', 'seq_start']

    PRICE_COLS = ['price_min', 'price_max']
    TARGET_COL = 'price_min'

    def __init__(self, csv_filepath_or_df):
        super(PriceHistory, self).__init__()

        self.orig_df = pd.read_csv(csv_filepath_or_df, index_col=0, encoding='utf-8',
                                   quoting=csv.QUOTE_ALL) if isinstance(csv_filepath_or_df, basestring) and isfile(
            csv_filepath_or_df) else csv_filepath_or_df

        self.df = self.orig_df.drop(labels=self.SPECIAL_COLS, axis=1)

    def extractSequenceByLocation(self, iloc):
        return super(PriceHistory, self).extractSequenceByLocation(self.orig_df, iloc)

    # def extractAllSequences(self):
    #     return [self.extractSequenceByLocation(iloc=ii) for ii in xrange(len(self.orig_df))]

    def extractAllSequences(self):
        return super(PriceHistory, self).extractAllSequences(self.orig_df)


if __name__ == "__main__":
    print "DONE"
