# -*- coding: UTF-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import csv
import types


def is_static_method(method):
    return isinstance(method, types.FunctionType)


class PriceHistoryTrait(object):
    @staticmethod
    def extractSequenceByLocation(df, iloc):
        cur_item = df.iloc[iloc]
        seq_start = int(cur_item['seq_start'])
        seq_len = int(cur_item['sequence_length'])
        return cur_item.iloc[seq_start:seq_start + seq_len]

    def extractAllSequences(self, df):
        if is_static_method(self.extractSequenceByLocation):
            return [self.extractSequenceByLocation(df=df, iloc=ii) for ii in xrange(len(df))]
        else:
            return [self.extractSequenceByLocation(iloc=ii) for ii in xrange(len(df))]


if __name__ == "__main__":
    print "DONE"
