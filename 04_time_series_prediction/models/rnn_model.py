from __future__ import division
import numpy as np
from mylibs.py_helper import merge_dicts


class RnnModel(object):
    @staticmethod
    def trainEpoch(batch_size, state_size, sess, data_provider,
                   inputs,
                   targets,
                   train_step,
                   error,
                   init_state,
                   last_state,
                   accuracy=None,
                   extraFeedDict=None,
                   onlyLastErrorAndAcc=False):
        if extraFeedDict is None:
            extraFeedDict = {}

        train_error = 0.
        train_accuracy = 0.
        # train_error = []
        # train_accuracy = []

        num_batches = data_provider.num_batches

        def zeroState():
            return np.zeros(shape=(batch_size, state_size))

        cur_state = zeroState()

        for step, ((input_batch, target_batch), segmentPartCounter) in enumerate(data_provider):
            feed_dict = merge_dicts({inputs: input_batch,
                                     targets: target_batch,
                                     init_state: cur_state}, extraFeedDict)

            if accuracy is None:
                cur_state, _, batch_error = sess.run([last_state, train_step, error], feed_dict=feed_dict)
            else:
                cur_state, _, batch_error, batch_acc = sess.run([last_state, train_step, error, accuracy],
                                                                feed_dict=feed_dict)

            if (segmentPartCounter + 1) % data_provider.segment_part_count == 0:
                cur_state = zeroState()

                if onlyLastErrorAndAcc:  # only care about the last output of the training error and acc of the rnn
                    train_error += batch_error
                    if accuracy is not None:
                        train_accuracy += batch_acc

            if not onlyLastErrorAndAcc:  # care about all outputs of the training error and acc of the rnn
                train_error += batch_error
                if accuracy is not None:
                    train_accuracy += batch_acc

        train_error /= num_batches
        train_accuracy /= num_batches

        if accuracy is None:
            return train_error
        else:
            return train_error, train_accuracy


class DynRnnModel(object):
    def _initDynRnnModel(self, symbol=0):
        self.__symbol = symbol

    @staticmethod
    def trainEpoch(sess,
                   train_data,
                   inputs,
                   targets,
                   train_step,
                   error,
                   sequence_lens,
                   sequence_len_mask,
                   accuracy=None,
                   extraFeedDict=None):
        if extraFeedDict is None:
            extraFeedDict = {}

        train_error = 0.
        train_accuracy = 0.

        num_batches = train_data.num_batches

        for step, (input_batch, target_batch, sequence_lengths, seqlen_mask) in enumerate(train_data):
            # self.verifyInputDoesNotContainSymbol(batch=input_batch)

            feed_dict = merge_dicts({inputs: input_batch,
                                     targets: target_batch,
                                     sequence_lens: sequence_lengths,
                                     sequence_len_mask: seqlen_mask,
                                     }, extraFeedDict)

            if accuracy is None:
                _, batch_error = sess.run([train_step, error], feed_dict=feed_dict)
            else:
                _, batch_error, batch_acc = sess.run([train_step, error, accuracy], feed_dict=feed_dict)
                train_accuracy += batch_acc

            train_error += batch_error

        train_error /= num_batches
        train_accuracy /= num_batches

        if accuracy is None:
            return train_error
        else:
            return train_error, train_accuracy

    def verifyInputDoesNotContainSymbol(self, batch):
        assert batch[batch == self.__symbol].size == 0

    def getSequenceLengths(self, batch):
        return [self.getSequenceLen(seq=seq) for seq in batch]

    def getSequenceLen(self, seq):
        """zero padding is the default"""
        return np.argwhere(seq == self.__symbol).T[0][0]

    @staticmethod
    def getSequenceLen_alt(seq):
        return len(np.trim_zeros(seq.T[0], trim='b'))
