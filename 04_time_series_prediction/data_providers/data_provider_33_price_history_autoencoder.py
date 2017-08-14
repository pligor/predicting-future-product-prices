# -*- coding: utf-8 -*-
"""Data providers.
This module provides classes for loading datasets and iterating over batches of
data points.
"""
from __future__ import division

import numpy as np
from nn_io.data_provider import CrossValDataProvider
from collections import OrderedDict


class PriceHistoryAutoEncDataProvider(CrossValDataProvider):  # , SubSampleableDataProvider):
    """Data provider which provides attributes of both the price history and the static mobile attrs of an instance"""
    EXPECTED_SETS = ['train', 'test']

    EOS_TOKEN_DEFAULT = 0.  # float(1e4)
    DATALIST_INPUTS_IND = 0
    DATALIST_SEQLENS_IND = 2
    TS_IND = 0

    def __init__(self,
                 # forced to set some default values because of some pythonic at some other place in the code. do NOT remove them
                 npz_path=None,
                 batch_size=None,
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

        complete_npz_path = npz_path + '_' + self.which_set + '.npz'

        # arr = self.subsample_dic(np.load(npz_path), target_size_or_fraction=target_size_or_fraction, random_state=rng)
        dic = np.load(complete_npz_path)  # , target_size_or_fraction=target_size_or_fraction, random_state=rng)

        # currently this is of static length therefore we do not care of providing more of the stored information (sku_ids, seq lens and seq masks)
        inputs = dic['inputs']
        decoder_extra_inputs = dic['extra_inputs']
        seqlens = dic['sequence_lengths']
        seqmasks = dic['sequence_masks']

        dataset_len = len(inputs)
        assert dataset_len % batch_size == 0, "batch size to be chosen so that it divides the dataset exactly " \
                                              "dataset len {}, batch_size {}".format(dataset_len, batch_size)

        self.input_len = len(inputs[0])
        # assert input_len % trunc_backprop_len == 0, \
        #     "the truncated backprop len must divide the length of the input sequence exactly"

        feature_len = inputs[0].shape[1]
        assert feature_len == 7

        # pass the loaded data to the parent class __init__
        super(PriceHistoryAutoEncDataProvider, self).__init__(
            datalist=[inputs, decoder_extra_inputs, seqlens, seqmasks],
            batch_size=batch_size, max_num_batches=max_num_batches,
            shuffle_order=shuffle_order, rng=rng, indices=indices)

    def next(self):
        """
        This is why this class is called Dummy, because we are providing the decoder inputs to the decoder rnn from our
        targets which is cheating
        """
        inputs_batch, decoder_extra_inputs_batch, seqlens_batch, seqmasks_batch = super(PriceHistoryAutoEncDataProvider,
                                                                                        self).next()

        # final_targets = np.pad(targets_batch, ((0, 0), (0, 1)), mode='constant',
        #                        constant_values=self.__eos_token) if self.with_EOS else targets_batch

        # so here we have the date information for the current output, obviously this should be the same date as the output
        # But how should we treat the case with the EOS (end of sequence) ? What kind of date information will be
        # provided for that case?
        # It could be something invalid like setting everything to -1 OR it could make sense to just be the next day
        # since we are not currently working with the EOS solution we are going to not bother with this part for now

        # final_dec_extra_ins = np.pad(decoder_extra_inputs_batch, ((0, 0), (0, 1), (0, 0)), mode='constant',
        #                              constant_values=-1
        #                              ) if self.with_EOS else decoder_extra_inputs_batch
        final_dec_extra_ins = decoder_extra_inputs_batch

        # Note that targets correspond from the second input of the decoder onwards
        # so it is our responsibility to feed the first input correctly. In other words
        # all_targets_but_last = targets_batch[:, :-1]
        # dec_inputs = np.pad(targets_batch, ((0, 0), (1, 0)), mode='constant',
        #                     constant_values=self.__eos_token) if self.with_EOS else all_targets_but_last
        # dec_inputs = targets_batch if self.with_EOS else all_targets_but_last
        # final_dec_inputs = np.reshape(dec_inputs, newshape=dec_inputs.shape + (1,))

        return inputs_batch, final_dec_extra_ins, seqlens_batch, seqmasks_batch

    @property
    def inputs(self):
        return self.datalist[self.DATALIST_INPUTS_IND]

    @property
    def seqlens(self):
        return self.datalist[self.DATALIST_SEQLENS_IND]

    @property
    def targets(self):
        return self.datalist[self.DATALIST_INPUTS_IND][:, :, self.TS_IND]

    # def get_inputs_dict(self):
    #     values = self.datalist[self.DATALIST_INPUTS_IND]
    #     keys = self.current_order
    #     return OrderedDict(zip(keys, values))

    def get_targets_dict(self):
        values = self.datalist[self.DATALIST_INPUTS_IND][:, :, self.TS_IND]
        keys = self.current_order
        return OrderedDict(zip(keys, values))

    def get_targets_dict_trimmed(self):
        targets_dict_trimmed = OrderedDict()

        for seqlen, (key, targets) in zip(self.seqlens, self.get_targets_dict().iteritems()):
            targets_dict_trimmed[key] = targets[:seqlen]

        return targets_dict_trimmed
