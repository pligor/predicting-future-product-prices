# -*- coding: UTF-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import csv
import types


def is_static_method(method):
    return isinstance(method, types.FunctionType)


class PriceHistoryTrait(object):
    def extractSequenceByLocation(self, df, iloc):
        return self.extractSequence(row=df.iloc[iloc])

    def extractAllSequences(self, df):
        # if is_static_method(self.extractSequenceByLocation):
        return [self.extractSequenceByLocation(df=df, iloc=ii) for ii in xrange(len(df))]
        # else:
        #     return [self.extractSequenceByLocation(iloc=ii) for ii in xrange(len(df))]

    @staticmethod
    def extractSequence(row):
        seq_start = int(row['seq_start'])
        seq_len = int(row['sequence_length'])
        return row.iloc[seq_start:seq_start + seq_len]
