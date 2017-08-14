from __future__ import division

import numpy as np
import tensorflow as tf

from data_providers.price_history_varlen_data_provider import PriceHistoryVarLenDataProvider
from models.predictions_gatherer import PredictionsGathererVarLen
from mylibs.jupyter_notebook_helper import DynStats, getRunTime
# from model_selection.cross_validator import CrossValidator
from interfaces.neural_net_model_interface import NeuralNetModelInterface
from tensorflow.contrib import rnn
from tensorflow.contrib import learn
from rnn_model import DynRnnModel
from cost_functions.huber_loss import huberLoss
from mylibs.tf_helper import tfMSE


class PriceHistoryRnnVarlen(PredictionsGathererVarLen, NeuralNetModelInterface, DynRnnModel):
    """https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html"""

    FEATURE_LEN = 1

    # SERIES_MAX_LEN = 60
    # TARGET_LEN = 30

    def __init__(self, rng, dtype, config):
        # super(ShiftFunc, self).__init__(random_state=rng, data_provider_class=ProductPairsDataProvider)
        super(PriceHistoryRnnVarlen, self).__init__()

        self.rng = rng
        self.dtype = dtype
        self.config = config
        self._initDynRnnModel()

        self.train_data = None
        self.valid_data = None
        self.init = None
        self.error = None
        # self.accuracy = None
        self.predictions = None
        self.inputs = None
        self.targets = None
        self.train_step = None
        self.last_state = None
        self.sequence_lens = None
        self.sequence_len_mask = None

    class COST_FUNCS(object):
        HUBER_LOSS = "huberloss"
        MSE = 'mse'

    class RNN_CELLS(object):
        BASIC_RNN = tf.contrib.rnn.BasicRNNCell  # "basic_rnn"

        @staticmethod
        def LSTM(num_units):
            return tf.contrib.rnn.BasicLSTMCell(num_units=num_units, state_is_tuple=False)  # "lstm"

        GRU = tf.contrib.rnn.GRUCell  # "gru"

    def run(self, epochs, state_size, npz_path, series_max_len, target_len, batch_size, preds_gather_enabled=True,
            cost_func=COST_FUNCS.HUBER_LOSS, rnn_cell=RNN_CELLS.BASIC_RNN):
        graph = self.getGraph(batch_size=batch_size, state_size=state_size, rnn_cell=rnn_cell,
                              verbose=False, series_max_len=series_max_len, target_len=target_len, cost_func=cost_func)

        dp = PriceHistoryVarLenDataProvider(filteringSeqLens=lambda xx: xx >= target_len, npz_path=npz_path,
                                            batch_size=batch_size)

        preds_dp = PriceHistoryVarLenDataProvider(filteringSeqLens=lambda xx: xx >= target_len, npz_path=npz_path,
                                                  batch_size=batch_size,
                                                  shuffle_order=False) if preds_gather_enabled else None

        return self.train_validate(train_data=dp, valid_data=None, graph=graph, epochs=epochs,
                                   target_len=target_len, preds_dp=preds_dp,
                                   batch_size=batch_size, preds_gather_enabled=preds_gather_enabled)

    def train_validate(self, train_data, valid_data, **kwargs):
        graph = kwargs['graph']
        epochs = kwargs['epochs']  # if 'epochs' in kwargs.keys() else 10
        target_len = kwargs['target_len']
        batch_size = kwargs['batch_size']
        preds_dp = kwargs['preds_dp'] if 'preds_dp' in kwargs.keys() else None
        verbose = kwargs['verbose'] if 'verbose' in kwargs.keys() else True
        preds_gather_enabled = kwargs['preds_gather_enabled'] if 'preds_gather_enabled' in kwargs.keys() else True

        if verbose:
            print "epochs: %d" % epochs

        # min_error = float("inf")

        with tf.Session(graph=graph, config=self.config) as sess:
            sess.run(self.init)  # sess.run(tf.initialize_all_variables())

            dynStats = DynStats(validation=False, accuracy=False)

            for epoch in range(epochs):
                train_error, runTime = getRunTime(
                    lambda:
                    self.trainEpoch(
                        inputs=self.inputs,
                        targets=self.targets,
                        sess=sess,
                        train_data=train_data,
                        train_step=self.train_step,
                        error=self.error,
                        sequence_lens=self.sequence_lens,
                        sequence_len_mask=self.sequence_len_mask
                    )
                )

                # if (epoch + 1) % 1 == 0:
                #     valid_error, valid_accuracy = validateEpoch(
                #         inputs=self.inputs,
                #         targets=self.targets,
                #         sess=sess,
                #         valid_data=valid_data,
                #         error=self.error,
                #         accuracy=self.accuracy,
                #         extraFeedDict={self.is_training: False},
                #         # keep_prob_keys=[self.keep_prob_input, self.keep_prob_hidden]
                #     )
                # valid_error, valid_accuracy = 0, 0

                # if train_error < min_error:
                #     if predictions_gathering_enabled:
                #         preds_dict, _ = self.getPredictions(target_len=target_len, batch_size=batch_size,
                #                                              data_provider=preds_dp, sess=sess)
                #     min_error = train_error

                if verbose:
                    # print 'EndEpoch%02d(%.3f secs):err(train)=%.4f,acc(train)=%.2f,err(valid)=%.2f,acc(valid)=%.2f, ' % \
                    #       (epoch + 1, runTime, train_error, train_accuracy, valid_error, valid_accuracy)
                    print 'End Epoch %02d (%.3f secs): err(train) = %.4f' % (epoch + 1, runTime, train_error)

                dynStats.gatherStats(train_error)

            predictions_dict = self.getPredictions(
                target_len=target_len, batch_size=batch_size, data_provider=preds_dp, sess=sess)[
                0] if preds_gather_enabled else None

        if verbose:
            print

        if preds_gather_enabled:
            return dynStats, predictions_dict
        else:
            return dynStats

    def getGraph(self,
                 batch_size,
                 state_size,
                 series_max_len,
                 target_len,
                 feature_len=FEATURE_LEN,
                 # learningRate=0.3,  # 1e-3,  # default of Adam is 1e-3
                 learningRate=1e-3,  # default of Adam is 1e-3
                 lamda2=1e-2,
                 cost_func=COST_FUNCS.HUBER_LOSS,
                 rnn_cell=RNN_CELLS.BASIC_RNN,
                 verbose=True):
        # momentum = 0.5
        # tf.reset_default_graph() #kind of redundant statement
        if verbose:
            # print "lamda2: %f" % lamda2
            print "learning rate: %f" % learningRate

        graph = tf.Graph()  # create new graph

        with graph.as_default():
            with tf.name_scope('data'):
                inputs = tf.placeholder(dtype=self.dtype,
                                        shape=(batch_size, series_max_len, feature_len),
                                        name="batch_x_placeholder")

                targets = tf.placeholder(dtype=self.dtype,
                                         shape=(batch_size, target_len),
                                         name="batch_y_placeholder")

                sequence_lens = tf.placeholder(tf.int32, shape=(batch_size,), name="sequence_lens_placeholder")

                sequence_len_mask = tf.placeholder(tf.int32, shape=(batch_size, series_max_len),
                                                   name="sequence_len_mask_placeholder")

            # with tf.name_scope('inputs'):
            #     inputs_series = tf.unstack(inputs, axis=1) # unpack matrix into 1 dim array
            #     # inputs_series = tf.split(inputs, truncated backprop len, axis=1)
            #     if verbose:
            #         print len(inputs_series)
            #         print inputs_series[0]  # shape: (5,)
            #         print

            with tf.name_scope('rnn_layer'):
                # note here that we are sharing the variable of init state so that we will not have a different state for
                # each different batch input
                init_single_state = tf.get_variable('init_state', [1, state_size],
                                                    initializer=tf.constant_initializer(0.0, dtype=self.dtype))
                init_state = tf.tile(init_single_state, [batch_size, 1])

                # [batch_size, max_time]
                # [5, 30, 1] #so we MUST KNOW the maximum length, here is 30, 5 is batch size, 1 is feature len
                rnn_outputs, last_state = tf.nn.dynamic_rnn(cell=rnn_cell(num_units=state_size),
                                                            inputs=inputs,
                                                            initial_state=init_state,
                                                            dtype=self.dtype,
                                                            sequence_length=sequence_lens)
                if verbose:
                    print "rnn_outputs:"
                    print rnn_outputs
                    print

            with tf.name_scope('gathering'):
                # we are gathering only the last outputs
                first_dim_inds = np.repeat(np.arange(batch_size), target_len).reshape(batch_size, target_len).astype(
                    np.int32)
                # first_dim_inds_tf = tf.constant(first_dim_inds, dtype=tf.int32)
                sec_dim_inds_tf = tf.transpose(tf.stack([sequence_lens - ii for ii in range(target_len, 0, -1)]))
                inds_tf = tf.stack((first_dim_inds, sec_dim_inds_tf), axis=2)

                # last_rnn_output = tf.gather_nd(rnn_outputs, tf.pack([tf.range(batch_size), seqlen - 1], axis=1))
                rnn_gathered_outs = tf.gather_nd(rnn_outputs, inds_tf)
                # which is the Tensorflow equivalent of numpy's rnn_outputs[range(30), seqlen-1, :]

                if verbose:
                    print rnn_gathered_outs
                    print

            gathered_outs_flat = tf.reshape(rnn_gathered_outs, (-1, state_size), name="flattening")
            if verbose:
                print gathered_outs_flat

            with tf.name_scope('readout_layer'):
                target_feature_len = 1  # this is considered static, because we are always trying to predict the price of a single product
                WW = tf.Variable(self.rng.randn(state_size, target_feature_len), dtype=self.dtype, name='WW')
                bb = tf.Variable(np.zeros(target_feature_len), dtype=self.dtype, name='bb')
                readout = tf.matmul(gathered_outs_flat, WW) + bb

                if verbose:
                    print readout
                    print

            with tf.name_scope('predictions'):
                predictions = tf.reshape(readout, shape=(batch_size, target_len))
                if verbose:
                    print predictions
                    print

            # with tf.name_scope('labels'):
            #     labels = tf.reshape(targets, shape=(-1, feature_len))
            #
            #     if verbose:
            #         print labels
            #         print

            with tf.name_scope('error'):
                if cost_func == self.COST_FUNCS.HUBER_LOSS:
                    losses = huberLoss(y_true=targets, y_pred=predictions)  # both have shape: (batch_size, target_len)
                elif cost_func == self.COST_FUNCS.MSE:
                    losses = tfMSE(outputs=predictions, targets=targets)
                else:
                    raise Exception("invalid or non supported cost function")


                #BECAREFUL: we have not applied the mask here because the target len is steady and therefore
                # all items play a role in the losses

                if verbose:
                    print losses
                    print

                # Here it is impossible to have losses that do not belong in the sequence because we are keeping only the
                # last outputs of the rnn which are equal to the target len which is always smaller than the sequence len
                control_dep = tf.assert_greater_equal(sequence_lens, target_len)
                with tf.control_dependencies([control_dep]):  # assert that target len < of all sequence len
                    # , axis=, no axis because we are summing across all target len and for all batches equally
                    error = tf.reduce_mean(losses)

                    if verbose:
                        print error
                        print

            # with tf.name_scope('accuracy'):
            #     accuracy = error

            with tf.name_scope('training_step'):
                train_step = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(error)

            init = tf.global_variables_initializer()

        self.init = init
        self.inputs = inputs
        self.targets = targets
        self.last_state = last_state
        self.sequence_lens = sequence_lens
        self.error = error
        # self.accuracy = accuracy
        self.predictions = predictions
        self.train_step = train_step
        self.sequence_len_mask = sequence_len_mask

        return graph
