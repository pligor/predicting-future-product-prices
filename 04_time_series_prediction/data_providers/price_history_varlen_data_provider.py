# -*- coding: utf-8 -*-
"""Data providers.
This module provides classes for loading datasets and iterating over batches of
data points.
"""
from __future__ import division

import csv
import os
import numpy as np
from nn_io.data_provider import UnifiedDataProvider  # , OneOfKDataProvider, CrossValDataProvider
import pandas as pd


class PriceHistoryVarLenDataProvider(UnifiedDataProvider):
    """Data provider. Note that the pairs are first WORSE (left features) and then BETTER (right features)"""
    EXPECTED_SETS = ['train']
    BATCH_SIZE = 5

    def __init__(self,
                 npz_path,
                 which_set='train', max_num_batches=-1, batch_size=BATCH_SIZE,
                 shuffle_order=True, rng=None, filteringSeqLens=None):
        """Create a new Price History data provider object.

        Args:
            which_set: One of 'train', 'valid' or 'test'. Determines which
                portion of the MNIST data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        # self.min_input_seq_len = min_input_seq_len
        # self.min_target_seq_len = min_target_seq_len

        assert which_set in self.EXPECTED_SETS, (
            'Expected which_set to be either {}. Got {}'.format(self.EXPECTED_SETS, which_set))

        self.which_set = which_set

        # TODO batch size to be chosen so that it divides the dataset

        arr = np.load(npz_path)

        inputs, targets, sequence_lens, seq_mask = arr['inputs'], arr['targets'], \
                                                   arr['sequence_lengths'], arr['sequence_masks']

        args = None if filteringSeqLens is None else np.argwhere(filteringSeqLens(sequence_lens)).flatten()

        self.inputs = inputs if args is None else inputs[args]
        self.targets = targets if args is None else targets[args]
        self.sequence_lengths = sequence_lens if args is None else sequence_lens[args]
        self.sequence_masks = seq_mask if args is None else seq_mask[args]

        # inputs = np.reshape(xx, (-1, truncated_backprop_len, xx.shape[1]))
        # targets = np.reshape(yy, (-1, truncated_backprop_len, yy.shape[1]))
        # inputs = inputs.astype(np.float32)
        # targets = targets.astype(np.int32)

        # pass the loaded data to the parent class __init__
        super(PriceHistoryVarLenDataProvider, self).__init__(
            datalist=[self.inputs, self.targets, self.sequence_lengths, self.sequence_masks],
            batch_size=batch_size, max_num_batches=max_num_batches,
            shuffle_order=shuffle_order, rng=rng)

    # @property
    # def csv_in(self):
    #     return self.__csv_in
    #
    # @csv_in.setter
    # def csv_in(self, value):
    #     self.__csv_in = value
