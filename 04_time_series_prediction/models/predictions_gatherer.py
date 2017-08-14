import numpy as np
from mylibs.py_helper import merge_dicts, factors
from collections import OrderedDict


class PredictionsGathererVarLen(object):
    def __init__(self):
        super(PredictionsGathererVarLen, self).__init__()
        # initialized by parents
        self.predictions = None
        self.error = None
        # self.accuracy = None
        self.inputs = None
        self.targets = None
        self.sequence_lens = None
        self.sequence_len_mask = None

    def getPredictions(self, target_len, sess, data_provider, batch_size, extraFeedDict=None):
        # batch_size = 97 #factors of 9991: [1, 103, 97, 9991]
        if extraFeedDict is None:
            extraFeedDict = {}

        # reasonable_batch_sizes = [ff for ff in factors(data_len) if 50 <= ff <= 100]
        # if len(reasonable_batch_sizes) == 0:
        #     raise Exception(
        #         "could not find a reasonable batch size from the factors of this data length: {]".format(data_len))
        #
        # batch_size = reasonable_batch_sizes[0]
        # data_provider = get_data_provider_by_batch_size(batch_size)
        assert data_provider.data_len % batch_size == 0

        total_error = 0.
        # total_accuracy = 0.

        instances_order = data_provider.current_order

        all_predictions = np.zeros(shape=(data_provider.data_len, target_len))

        for step, (input_batch, target_batch, sequence_lengths, seqlen_mask) in enumerate(data_provider):
            feed_dict = merge_dicts({self.inputs: input_batch,
                                     self.targets: target_batch,
                                     self.sequence_lens: sequence_lengths,
                                     self.sequence_len_mask: seqlen_mask,
                                     }, extraFeedDict)

            batch_error, batch_predictions = sess.run([self.error, self.predictions], feed_dict=feed_dict)

            all_predictions[step * batch_size: (step + 1) * batch_size, :] = batch_predictions
            assert np.all(instances_order == data_provider.current_order)

            total_error += batch_error

        num_batches = data_provider.num_batches

        total_error /= num_batches
        # total_accuracy /= num_batches

        assert np.all(all_predictions != 0)  # all predictions are expected to be something else than absolute zero

        preds_dict = OrderedDict(zip(instances_order, all_predictions))

        return preds_dict, total_error  # , total_accuracy
