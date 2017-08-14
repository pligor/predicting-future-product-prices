# -*- coding: utf-8 -*-
"""Data providers.
This module provides classes for loading datasets and iterating over batches of
data points.
"""
from __future__ import division

import numpy as np
from nn_io.data_provider import \
    CrossValDataProvider, \
    SubSampleableDataProvider  # UnifiedDataProvider  # , RnnStaticLenDataProvider  # , OneOfKDataProvider,
from collections import OrderedDict


class PriceHistorySeq2SeqDataProvider(CrossValDataProvider):  # , SubSampleableDataProvider):
    """Data provider which is dummy because we also provider the decoder inputs which in reality are unknowns"""
    EXPECTED_SETS = ['train', 'test']

    BATCH_SIZE = 50  # 47
    EOS_TOKEN_DEFAULT = 0.  # float(1e4)
    NPZ_PATH = 'train.npz'  # typically invalid
    DATALIST_TARGETS_IND = 1

    def __init__(self,
                 npz_path=NPZ_PATH,
                 batch_size=BATCH_SIZE,
                 eos_token=None, with_EOS=True,
                 which_set='train', max_num_batches=-1,
                 shuffle_order=True, rng=None, indices=None,
                 # target_size_or_fraction=None
                 ):
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
        self.__eos_token = self.EOS_TOKEN_DEFAULT if eos_token is None else eos_token  # 0. # recall that the inputs are normalized to zero but for the first element of input series, not targets
        self.with_EOS = with_EOS

        assert which_set in self.EXPECTED_SETS, (
            'Expected which_set to be either {}. Got {}'.format(self.EXPECTED_SETS, which_set))

        self.which_set = which_set

        # arr = self.subsample_dic(np.load(npz_path), target_size_or_fraction=target_size_or_fraction, random_state=rng)
        arr = np.load(npz_path)  # , target_size_or_fraction=target_size_or_fraction, random_state=rng)

        # currently this is of static length therefore we do not care of providing more of the stored information (sku_ids, seq lens and seq masks)
        inputs, targets = arr['inputs'], arr['targets']

        dataset_len = len(inputs)
        assert dataset_len % batch_size == 0, "batch size to be chosen so that it divides the dataset exactly " \
                                              "dataset len {}, batch_size {}".format(dataset_len, batch_size)

        input_len = len(inputs[0])
        # assert input_len % trunc_backprop_len == 0, \
        #     "the truncated backprop len must divide the length of the input sequence exactly"

        target_len = len(targets[0])
        feature_len = inputs[0].shape[1]
        assert feature_len == 1, "keep this until you advance it to more complex scenarios"

        self.inputs = inputs
        self.targets = targets

        # inputs = np.reshape(xx, (-1, trunc_backprop_len, xx.shape[1]))
        # targets = np.reshape(yy, (-1, trunc_backprop_len, yy.shape[1]))

        # pass the loaded data to the parent class __init__
        super(PriceHistorySeq2SeqDataProvider, self).__init__(
            datalist=[self.inputs, self.targets],
            batch_size=batch_size, max_num_batches=max_num_batches,
            shuffle_order=shuffle_order, rng=rng, indices=indices)

    def next(self):
        """
        This is why this class is called Dummy, because we are providing the decoder inputs to the decoder rnn from our
        targets which is cheating
        """
        inputs_batch, targets_batch = super(PriceHistorySeq2SeqDataProvider, self).next()

        final_targets = np.pad(targets_batch, ((0, 0), (0, 1)), mode='constant',
                               constant_values=self.__eos_token) if self.with_EOS else targets_batch

        # Note that targets correspond from the second input of the decoder onwards
        # so it is our responsibility to feed the first input correctly. In other words
        # all_targets_but_last = targets_batch[:, :-1]
        # dec_inputs = np.pad(targets_batch, ((0, 0), (1, 0)), mode='constant',
        #                     constant_values=self.__eos_token) if self.with_EOS else all_targets_but_last
        # dec_inputs = targets_batch if self.with_EOS else all_targets_but_last
        #
        # final_dec_inputs = np.reshape(dec_inputs, newshape=dec_inputs.shape + (1,))

        return inputs_batch, final_targets

    def get_targets_dict(self):
        targets = self.datalist[self.DATALIST_TARGETS_IND]
        keys = self.current_order
        return OrderedDict(zip(keys, targets))
