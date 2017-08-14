# -*- coding: utf-8 -*-
"""Data providers.
This module provides classes for loading datasets and iterating over batches of
data points.
"""
from __future__ import division

import numpy as np
from nn_io.data_provider import UnifiedDataProvider #RnnStaticLenDataProvider  # , OneOfKDataProvider, CrossValDataProvider

class PriceHistoryFullLenDataProvider(UnifiedDataProvider):
    """Data provider"""
    EXPECTED_SETS = ['train']
    # BATCH_SIZE = 47

    def __init__(self,
                 npz_path, batch_size,
                 which_set='train', max_num_batches=-1,
                 shuffle_order=True, rng=None):
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

        arr = np.load(npz_path)

        inputs, targets = arr['inputs'], arr['targets']

        assert len(inputs) % batch_size == 0, "batch size to be chosen so that it divides the dataset exactly"

        self.inputs = inputs
        self.targets = targets

        # pass the loaded data to the parent class __init__
        super(PriceHistoryFullLenDataProvider, self).__init__(
            datalist=[self.inputs, self.targets],
            batch_size=batch_size, max_num_batches=max_num_batches,
            shuffle_order=shuffle_order, rng=rng)
