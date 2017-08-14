# -*- coding: utf-8 -*-
"""Data providers.
This module provides classes for loading datasets and iterating over batches of
data points.
"""
from __future__ import division

import os
import numpy as np
import pandas as pd
from nn_io.data_provider import UnifiedDataProvider, OneOfKDataProvider, CrossValDataProvider
from sklearn.model_selection import StratifiedShuffleSplit


class BinaryShifterVarLenDataProvider(UnifiedDataProvider):
    """Data provider. Note that the pairs are first WORSE (left features) and then BETTER (right features)"""
    NUM_CLASSES = 2
    EXPECTED_SETS = ['train']
    BATCH_SIZE = 5

    def __init__(self, N_instances, series_max_len, echo_step,
                 which_set='train', max_num_batches=-1, batch_size=BATCH_SIZE, num_classes=NUM_CLASSES,
                 shuffle_order=True, rng=None):
        """Create a new MNIST data provider object.

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
        assert series_max_len // batch_size == series_max_len / batch_size

        assert which_set in self.EXPECTED_SETS, (
            'Expected which_set to be either {}. Got {}'.format(self.EXPECTED_SETS, which_set))

        self.which_set = which_set
        self.num_classes = num_classes

        inputs, targets, sequence_lens, seq_mask = self.generate_dataset(series_max_len=series_max_len,
                                                                         echo_step=echo_step,
                                                                         random_state=rng, num_instances=N_instances,
                                                                         num_classes=num_classes)

        # inputs = np.reshape(xx, (-1, truncated_backprop_len, xx.shape[1]))
        # targets = np.reshape(yy, (-1, truncated_backprop_len, yy.shape[1]))
        # inputs = inputs.astype(np.float32)
        # targets = targets.astype(np.int32)

        # pass the loaded data to the parent class __init__
        super(BinaryShifterVarLenDataProvider, self).__init__(
            datalist=[inputs, targets, sequence_lens, seq_mask],
            batch_size=batch_size, max_num_batches=max_num_batches,
            shuffle_order=shuffle_order, rng=rng)

    @staticmethod
    def generate_dataset(num_instances, series_max_len, echo_step, random_state, num_classes, num_features=1):
        XX = np.empty((0, series_max_len, num_features))
        YY = np.empty((0, series_max_len))
        sequence_lens = []
        seq_mask = np.empty((0, series_max_len))

        for ii in xrange(num_instances):
            xx, yy = BinaryShifterVarLenDataProvider.generate_instance(series_max_len=series_max_len,
                                                                       echo_step=echo_step,
                                                                       random_state=random_state,
                                                                       num_classes=num_classes)
            xx_len = len(xx)
            sequence_lens.append(xx_len)

            # build current mask with zeros and ones
            cur_mask = np.zeros(series_max_len)
            cur_mask[:xx_len] = 1  # only the valid firsts should have the value of one

            xx_padded = np.pad(xx, ((0, series_max_len - len(xx)), (0, 0)), mode='constant', constant_values=0.)
            yy_padded = np.pad(yy, (0, series_max_len - len(yy)), mode='constant', constant_values=0.)
            assert len(xx_padded) == series_max_len and len(yy_padded) == series_max_len

            XX = np.vstack((XX, xx_padded[np.newaxis]))
            YY = np.vstack((YY, yy_padded[np.newaxis]))
            seq_mask = np.vstack((seq_mask, cur_mask[np.newaxis]))

        return XX, YY, np.array(sequence_lens), seq_mask

    @staticmethod
    def generate_instance(num_classes, series_max_len, echo_step, random_state=None):
        """."""

        random_state = np.random if random_state is None else random_state
        # 0,1 50K samples
        # xx = np.array()

        randomly_generated_len = random_state.choice(series_max_len) + 1

        xx = random_state.choice(num_classes,
                                 randomly_generated_len)  # p=[.5, .5] is useless because default is uniform distribution

        # xx = np.arange(series_total_fixed_len) <<-- for testing purposes

        yy = np.roll(xx, shift=echo_step)  # pushes items to the right and then puts them at the beginning of the array
        yy[:echo_step] = 0  # we want to drop the first items, make them zero

        xx[xx == 0] = -1  # make them -1 in order to be able and pad them with zeros later

        # xx = xx.reshape((self.batch_size, -1))
        # yy = yy.reshape((self.batch_size, -1))
        xx = xx[np.newaxis].T  # here we have only one feature but in the general case we could have multiple features

        return xx, yy
