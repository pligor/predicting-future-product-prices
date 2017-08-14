# -*- coding: utf-8 -*-
"""Data providers.
This module provides classes for loading datasets and iterating over batches of
data points.
"""
from __future__ import division

import numpy as np
from nn_io.data_provider import \
    UnifiedDataProvider  # , RnnStaticLenDataProvider  # , OneOfKDataProvider, CrossValDataProvider


class PriceHistorySlidingWindowDataProvider(UnifiedDataProvider):
    """Data provider. Note that the pairs are first WORSE (left features) and then BETTER (right features)"""
    EXPECTED_SETS = ['train']
    DEFAULT_PRED_LEN = 1

    # BATCH_SIZE = 47

    def __init__(self,
                 npz_path, batch_size, trunc_backprop_len, pred_len=DEFAULT_PRED_LEN,
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
        super(PriceHistorySlidingWindowDataProvider, self).__init__(
            datalist=[self.inputs, self.targets],
            batch_size=batch_size, max_num_batches=max_num_batches,
            shuffle_order=shuffle_order, rng=rng)

        self.__init(trunc_backprop_len=trunc_backprop_len,
                    input_len=input_len, feature_len=feature_len, target_len=target_len, pred_len=pred_len)

    def __init(self, trunc_backprop_len, input_len, target_len, feature_len, pred_len):
        """segment count is into how many pieces does the inputs are splitted because of the truncated backprop length"""
        self.__full_seq_len = input_len + target_len
        self.__trunc_backprop_len = trunc_backprop_len
        self.__input_len = input_len
        self.__target_len = target_len
        self.__pred_len = pred_len
        self.__target_feature_len = 1  # the pricing history is being predicted

        self.feature_len = feature_len

        self.__part_count = self.__full_seq_len - self.__trunc_backprop_len  # - self.__pred_len

        # assert self.__part_count % (self.__trunc_backprop_len + self.__pred_len) == 0, \
        assert self.__part_count % self.__pred_len == 0, \
            "the prediction sequence length should fit nicely in the part count"

        self.__reset()

    def __reset(self):
        self.__counter = 0
        inputs_batch, targets_batch = super(PriceHistorySlidingWindowDataProvider, self).next()
        self.__inputs_batch = inputs_batch  # self.__reshape(batch=inputs_batch)
        self.__targets_batch = np.concatenate(
            (inputs_batch, targets_batch.reshape(self.batch_size, -1, self.__target_feature_len)), axis=1)
        self.__cur_preds_stream = np.empty(shape=(self.batch_size, 0, self.__target_feature_len))

    def next(self):
        """testing:
        full_len = 90
        aa = np.arange(full_len)
        pred_len = 1
        trunc_len = 60
        count = full_len - trunc_len #- pred_len
        print count % pred_len #+ trunc_len
        print count
        for xx in range(0, count, pred_len):
            print aa[xx:xx+trunc_len+pred_len]
        """
        # if self.__counter % self.__part_count == 0 or self.__counter >= self.__part_count:
        new_instance = False
        if self.__counter >= self.__part_count:
            self.__reset()
            new_instance = True
            # print "new_instance"

        input_slice = slice(self.__counter, self.__counter + self.__trunc_backprop_len)  # this is a safe slice
        target_slice = slice(self.__counter + self.__trunc_backprop_len,
                             self.__counter + self.__trunc_backprop_len + self.__pred_len)

        cur_input_batch = self.__inputs_batch[:, input_slice, :]
        cur_input_len = cur_input_batch.shape[1]

        # print cur_input_len, cur_input_batch.shape[1]

        if cur_input_len < self.__trunc_backprop_len:  # this is an indication that we have overflowed and continue with the predictions
            remain_len = self.__trunc_backprop_len - cur_input_len
            useful_preds = self.__cur_preds_stream[:, -remain_len:, :].reshape(self.batch_size, -1,
                                                                               self.__target_feature_len)
            cur_input_batch = np.concatenate((cur_input_batch, useful_preds), axis=1)
            assert cur_input_batch.shape[
                       1] == self.__trunc_backprop_len, \
                "don't forget to call stackPreds method, cur input batch shape {} and trunc backprop len {}".format(
                    cur_input_batch.shape, self.__trunc_backprop_len)

        cur_target_batch = self.__targets_batch[:, target_slice, :]

        # cur_counter = self.__counter

        self.__counter += self.__pred_len

        losses_mask = np.zeros(shape=cur_target_batch.shape).astype(np.float32)
        if target_slice.stop <= self.__input_len:
            count_targets_of_interest = 0  # account no piece
            # nothing else for losses mask
        elif target_slice.start >= self.__input_len:
            count_targets_of_interest = self.__pred_len  # account all pieces
            losses_mask = np.ones(shape=cur_target_batch.shape).astype(np.float32)
        else:
            count_targets_of_interest = target_slice.stop - self.__input_len
            losses_mask[:, self.__pred_len - count_targets_of_interest:, :] = 1

        return cur_input_batch, cur_target_batch, new_instance, count_targets_of_interest, losses_mask

    def stackPreds(self, new_preds):
        # self.__cur_preds_stream = np.hstack((self.__cur_preds_stream, new_preds))
        self.__cur_preds_stream = np.concatenate((self.__cur_preds_stream, new_preds), axis=1)
