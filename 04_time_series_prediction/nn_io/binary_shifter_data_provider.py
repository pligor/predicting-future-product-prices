# -*- coding: utf-8 -*-
"""Data providers.

This module provides classes for loading datasets and iterating over batches of
data points.
"""

import os
import numpy as np
import pandas as pd
from data_provider import DataProvider, OneOfKDataProvider, CrossValDataProvider, RnnStaticLenDataProvider
from sklearn.model_selection import StratifiedShuffleSplit


class BinaryShifterDataProvider(RnnStaticLenDataProvider, DataProvider):
    """Data provider. Note that the pairs are first WORSE (left features) and then BETTER (right features)"""
    NUM_CLASSES = 2
    EXPECTED_SETS = ['train']
    BATCH_SIZE = 5

    def __init__(self, N_instances, total_series_length, echo_step, truncated_backprop_len,
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
        assert total_series_length // batch_size // truncated_backprop_len == total_series_length / batch_size / truncated_backprop_len

        assert which_set in self.EXPECTED_SETS, (
            'Expected which_set to be either {}. Got {}'.format(self.EXPECTED_SETS, which_set))

        self._RnnStaticLen_init(batch_size=batch_size, truncated_backprop_len=truncated_backprop_len,
                                series_total_fixed_len=total_series_length,
                                process_targets=True,
                                dimensionality_of_point_in_time=1)

        self.which_set = which_set
        self.num_classes = num_classes

        inputs, targets = self.generate_dataset(series_total_fixed_len=total_series_length, echo_step=echo_step,
                                                random_state=rng, num_instances=N_instances)

        # inputs = np.reshape(xx, (-1, truncated_backprop_len, xx.shape[1]))
        # targets = np.reshape(yy, (-1, truncated_backprop_len, yy.shape[1]))
        # inputs = inputs.astype(np.float32)
        # targets = targets.astype(np.int32)

        # pass the loaded data to the parent class __init__
        super(BinaryShifterDataProvider, self).__init__(
            inputs=inputs, targets=targets, batch_size=batch_size, max_num_batches=max_num_batches,
            shuffle_order=shuffle_order, rng=rng)

    def generate_dataset(self, num_instances, series_total_fixed_len, echo_step, random_state, num_features=1):
        XX = np.empty((0, series_total_fixed_len, num_features))
        YY = np.empty((0, series_total_fixed_len, num_features))

        for ii in xrange(num_instances):
            xx, yy = self.__generate_instance(series_total_fixed_len=series_total_fixed_len, echo_step=echo_step,
                                              random_state=random_state)
            XX = np.vstack((XX, xx[np.newaxis]))
            YY = np.vstack((YY, yy[np.newaxis]))

        return XX, YY

    def __generate_instance(self, series_total_fixed_len, echo_step, random_state=None):
        """dim: (1,)"""

        random_state = np.random if random_state is None else random_state
        # 0,1 50K samples
        # xx = np.array()

        xx = random_state.choice(self.num_classes, series_total_fixed_len)
        xx[xx == 0] = -1  # make them -1
        # print xx
        # exit()

        # xx = np.arange(series_total_fixed_len) <<-- for testing purposes

        # p=[.5, .5] is useless because
        # default is uniform distribution

        # pushes items to the right and then puts them at the beginning of the array
        yy = np.roll(xx, shift=echo_step)

        yy[:echo_step] = 0  # we want to drop the first items, make them zero
        yy[yy == -1] = 0

        # xx = xx.reshape((self.batch_size, -1))
        # yy = yy.reshape((self.batch_size, -1))
        xx = xx[np.newaxis].T  # here we have only one feature but in the general case we could have multiple features
        yy = yy[np.newaxis].T  # here we have only one feature but in the general case we could have multiple features

        return xx, yy


if __name__ == '__main__':
    print BinaryShifterDataProvider.mro()


    class X(object):
        pass


    class Y(object):
        pass


    class Z(object):
        pass


    class A(X, Y):
        pass


    class B(Y, Z):
        pass


    class M(B, A, Z):
        pass


    # Output:
    # [<class '__main__.M'>, <class '__main__.B'>,
    # <class '__main__.A'>, <class '__main__.X'>,
    # <class '__main__.Y'>, <class '__main__.Z'>,
    # <class 'object'>]

    print(M.mro())
