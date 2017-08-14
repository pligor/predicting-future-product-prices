from __future__ import division

import numpy as np
import tensorflow as tf

from cost_functions.huber_loss import huberLoss
from data_providers.data_provider_11_price_history_dummy_seq2seq import PriceHistoryDummySeq2SeqDataProvider
from mylibs.jupyter_notebook_helper import DynStats, getRunTime
# from model_selection.cross_validator import CrossValidator
from interfaces.neural_net_model_interface import NeuralNetModelInterface
from tensorflow.contrib import rnn
from collections import OrderedDict
from mylibs.py_helper import merge_dicts
from mylibs.tf_helper import tfMSE


class PriceHistoryDummySeq2Seq(NeuralNetModelInterface):
    FEATURE_LEN = 1
    TARGET_FEATURE_LEN = 1

    def __init__(self, rng, dtype, config, with_EOS=False):
        # super(ShiftFunc, self).__init__(random_state=rng, data_provider_class=ProductPairsDataProvider)
        super(PriceHistoryDummySeq2Seq, self).__init__()

        self.rng = rng
        self.dtype = dtype
        self.config = config
        self.train_data = None
        self.valid_data = None
        self.init = None
        self.error = None
        self.inputs = None
        self.predictions = None
        self.targets = None
        self.train_step = None
        self.decoder_inputs = None

        # end of sequence token length is necessary for machine translation models, not sure for our model
        self.EOS_TOKEN_LEN = 1 if with_EOS else 0
        self.with_EOS = with_EOS

    class COST_FUNCS(object):
        HUBER_LOSS = "huberloss"
        MSE = 'mse'

    class RNN_CELLS(object):
        BASIC_RNN = tf.contrib.rnn.BasicRNNCell  # "basic_rnn"

        GRU = tf.contrib.rnn.GRUCell  # "gru"

    def run(self, npz_path, epochs, batch_size, num_units, input_len, target_len, preds_gather_enabled=True,
            cost_func=COST_FUNCS.HUBER_LOSS, rnn_cell=RNN_CELLS.BASIC_RNN):

        graph = self.getGraph(batch_size=batch_size, verbose=False, num_units=num_units, input_len=input_len,
                              rnn_cell=rnn_cell, target_len=target_len, cost_func=cost_func)

        train_data = PriceHistoryDummySeq2SeqDataProvider(npz_path=npz_path, batch_size=batch_size, rng=self.rng,
                                                          with_EOS=self.with_EOS)

        preds_dp = PriceHistoryDummySeq2SeqDataProvider(npz_path=npz_path, batch_size=batch_size, rng=self.rng,
                                                        shuffle_order=False, with_EOS=self.with_EOS
                                                        ) if preds_gather_enabled else None

        stats = self.train_validate(train_data=train_data, valid_data=None, graph=graph, epochs=epochs,
                                    preds_gather_enabled=preds_gather_enabled, preds_dp=preds_dp, batch_size=batch_size)

        return stats

    def train_validate(self, train_data, valid_data, **kwargs):
        graph = kwargs['graph']
        epochs = kwargs['epochs']
        batch_size = kwargs['batch_size']
        verbose = kwargs['verbose'] if 'verbose' in kwargs.keys() else True
        preds_dp = kwargs['preds_dp'] if 'preds_dp' in kwargs.keys() else None
        preds_gather_enabled = kwargs['preds_gather_enabled'] if 'preds_gather_enabled' in kwargs.keys() else True

        if verbose:
            print "epochs: %d" % epochs

        with tf.Session(graph=graph, config=self.config) as sess:
            sess.run(self.init)  # sess.run(tf.initialize_all_variables())

            dynStats = DynStats()

            for epoch in range(epochs):
                train_error, runTime = getRunTime(
                    lambda:
                    self.trainEpoch(
                        sess=sess,
                        data_provider=train_data,
                    )
                )

                # if (epoch + 1) % 1 == 0:
                # valid_error = validateEpoch(
                #     inputs=self.inputs,
                #     targets=self.targets,
                #     sess=sess,
                #     valid_data=valid_data,
                #     error=self.error,
                #     extraFeedDict={self.is_training: False},
                #     # keep_prob_keys=[self.keep_prob_input, self.keep_prob_hidden]
                # )

                if verbose:
                    # print 'EndEpoch%02d(%.3f secs):err(train)=%.4f,acc(train)=%.2f,err(valid)=%.2f,acc(valid)=%.2f, ' % \
                    #       (epoch + 1, runTime, train_error, train_accuracy, valid_error, valid_accuracy)
                    print 'End Epoch %02d (%.3f secs): err(train) = %.4f' % (epoch + 1, runTime, train_error)

                dynStats.gatherStats(train_error)

            predictions_dict = self.getPredictions(batch_size=batch_size, data_provider=preds_dp, sess=sess)[
                0] if preds_gather_enabled else None

        if verbose:
            print

        if preds_gather_enabled:
            return dynStats, predictions_dict
        else:
            return dynStats

    def getPredictions(self, sess, data_provider, batch_size, extraFeedDict=None):
        if extraFeedDict is None:
            extraFeedDict = {}

        assert data_provider.data_len % batch_size == 0

        total_error = 0.

        instances_order = data_provider.current_order

        target_len = data_provider.targets.shape[1]

        all_predictions = np.zeros(shape=(data_provider.data_len, target_len))

        for inst_ind, (input_batch, target_batch, dec_inputs) in enumerate(data_provider):
            cur_error, cur_preds = sess.run(
                [self.error, self.predictions],
                feed_dict=merge_dicts({self.inputs: input_batch,
                                       self.targets: target_batch,
                                       self.decoder_inputs: dec_inputs,
                                       }, extraFeedDict))

            assert np.all(instances_order == data_provider.current_order)

            all_predictions[inst_ind * batch_size: (inst_ind + 1) * batch_size, :] = cur_preds[:,
                                                                                     :-1] if self.with_EOS else cur_preds

            total_error += cur_error

        total_error /= data_provider.num_batches

        assert np.all(all_predictions != 0)  # all predictions are expected to be something else than absolute zero

        preds_dict = OrderedDict(zip(instances_order, all_predictions))

        return preds_dict, total_error

    def trainEpoch(self, sess, data_provider, extraFeedDict=None):
        if extraFeedDict is None:
            extraFeedDict = {}

        train_error = 0.

        num_batches = data_provider.num_batches

        for step, (input_batch, target_batch, dec_inputs) in enumerate(data_provider):
            feed_dic = merge_dicts({self.inputs: input_batch,
                                    self.targets: target_batch,
                                    self.decoder_inputs: dec_inputs,
                                    }, extraFeedDict)

            _, batch_error = sess.run([self.train_step, self.error], feed_dict=feed_dic)

            train_error += batch_error

        train_error /= num_batches

        return train_error

    def testing(self, graph, config):
        with tf.Session(graph=graph, config=config) as sess:
            sess.run(self.init)  # sess.run(tf.initialize_all_variables())

            preds, err = sess.run([self.predictions, self.error], feed_dict={
                self.inputs: np.arange(47 * 60 * 1).reshape((47, 60, 1)),
                self.targets: np.arange(47 * 30).reshape((47, 30)),
                self.decoder_inputs: np.arange(47 * 30 * 1).reshape((47, 30, 1)),
            })

        return preds, err

    def getGraph(self,
                     # TODO in this version we are building it full length and then we are going to improve it (trunc backprop len)
                     batch_size,
                     num_units,
                     input_len,
                     target_len,
                     rnn_cell=RNN_CELLS.BASIC_RNN,
                     cost_func=COST_FUNCS.HUBER_LOSS,
                     learningRate=1e-3,  # default of Adam is 1e-3
                     # lamda2=1e-2,
                     verbose=True):
        # momentum = 0.5
        # tf.reset_default_graph() #kind of redundant statement
        if verbose:
            # print "lamda2: %f" % lamda2
            print "learning rate: %f" % learningRate

        output_seq_len = target_len + self.EOS_TOKEN_LEN

        graph = tf.Graph()  # create new graph

        with graph.as_default():
            with tf.name_scope('data'):
                inputs = tf.placeholder(dtype=self.dtype,
                                        shape=(batch_size, input_len, self.FEATURE_LEN), name="inputs")

                targets = tf.placeholder(dtype=self.dtype, shape=(batch_size, output_seq_len), name="targets")

                # temporary to make it easy for ourselves the first time
                decoder_inputs = tf.placeholder(dtype=self.dtype,
                                                shape=(batch_size, output_seq_len,
                                                       self.TARGET_FEATURE_LEN))

            with tf.name_scope('inputs'):
                # unpack matrix into 1 dim array
                inputs_series = tf.unstack(inputs, axis=1)

                if verbose:
                    print len(inputs_series)
                    print inputs_series[0]  # shape: (5,)
                    print

            with tf.name_scope('encoder_rnn_layer'):
                # init_single_state = tf.get_variable('init_state', [1, num_units],
                #                                     initializer=tf.constant_initializer(0.0, dtype=self.dtype))
                # initial_state = tf.tile(init_single_state, [batch_size, 1])

                # don't really care for encoder outputs, but only for its final state
                # the encoder consumes all the input to get a sense of the trend of price history
                _, encoder_final_state = rnn.static_rnn(cell=rnn_cell(num_units=num_units),
                                                        inputs=inputs_series,
                                                        initial_state=None,
                                                        # TODO when using trunc backprop this should not be zero
                                                        dtype=self.dtype)

                if verbose:
                    print encoder_final_state
                    print

            with tf.variable_scope('decoder_rnn_layer'):
                decoder_inputs_series = tf.unstack(decoder_inputs, axis=1)
                if verbose:
                    print "decoder inputs series"
                    print type(decoder_inputs_series)
                    print len(decoder_inputs_series)
                    print decoder_inputs_series[0]
                    print

                # note that we use the same number of units for decoder here
                decoder_outputs, _ = rnn.static_rnn(cell=rnn_cell(num_units=num_units),
                                                    inputs=decoder_inputs_series,
                                                    initial_state=encoder_final_state,
                                                    dtype=self.dtype)

                if verbose:
                    print len(decoder_outputs)
                    print decoder_outputs[0]
                    print

                    # decoder_logits = tf.contrib.layers.linear(decoder_outputs, target_len)
                    # tf.contrib.layers.fully_connected(decoder_outputs)

            with tf.name_scope('decoder_outs'):
                dec_out_matrix = tf.reshape(tf.stack(decoder_outputs, axis=1), shape=(-1, num_units))

                if verbose:
                    print dec_out_matrix
                    print

            # with tf.name_scope('decoder_outs2'):
            #     stacked2 = tf.stack(decoder_outputs, axis=0)
            #
            #     if verbose:
            #         print stacked2
            #         print

            with tf.name_scope('readout_layer'):
                WW = tf.Variable(self.rng.randn(num_units, self.TARGET_FEATURE_LEN), dtype=self.dtype, name='weights')
                bb = tf.Variable(np.zeros(self.TARGET_FEATURE_LEN), dtype=self.dtype, name='bias')
                # readouts = [tf.matmul(output, WW) + bb for output in decoder_outputs]
                readouts = tf.add(tf.matmul(dec_out_matrix, WW), bb, name="readouts")

                if verbose:
                    # print len(readouts)
                    # print readouts[0]
                    print readouts
                    print

            with tf.name_scope('predictions'):
                #predictions = tf.reshape(tf.stack(readouts, axis=1), shape=(batch_size, output_seq_len))
                predictions = tf.reshape(readouts, shape=(batch_size, output_seq_len))
                if verbose:
                    print predictions
                    print

            with tf.name_scope('error'):
                if cost_func == self.COST_FUNCS.HUBER_LOSS:
                    losses = huberLoss(y_true=targets, y_pred=predictions)  # both have shape: (batch_size, target_len)
                elif cost_func == self.COST_FUNCS.MSE:
                    losses = tfMSE(outputs=predictions, targets=targets)
                else:
                    raise Exception("invalid or non supported cost function")

                if verbose:
                    print losses
                    print

                if self.with_EOS:  # fix error to exclude the EOS from the error calculation
                    mask = np.ones(shape=losses.get_shape())
                    mask[:, -1:] = 0
                    losses = losses * tf.constant(mask, dtype=tf.float32)

                error = tf.reduce_mean(losses)

                if verbose:
                    print error
                    print

            with tf.name_scope('training_step'):
                train_step = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(error)

            init = tf.global_variables_initializer()

        self.init = init
        self.inputs = inputs
        self.targets = targets

        self.error = error
        # self.is_training = is_training
        self.train_step = train_step
        self.predictions = predictions
        self.decoder_inputs = decoder_inputs

        # self.keep_prob_input = keep_prob_input
        # self.keep_prob_hidden = keep_prob_hidden

        return graph

    def getGraph_alt(self,
                 # TODO in this version we are building it full length and then we are going to improve it (trunc backprop len)
                 batch_size,
                 num_units,
                 input_len,
                 target_len,
                 rnn_cell=RNN_CELLS.BASIC_RNN,
                 cost_func=COST_FUNCS.HUBER_LOSS,
                 learningRate=1e-3,  # default of Adam is 1e-3
                 # lamda2=1e-2,
                 verbose=True):
        """
        PROOF THAT getGraph_alt approach is the same as getGraph
        with tf.Session() as sess:
            WW = tf.constant(np.arange(400*1).reshape(400,1), dtype=tf.float32)

            fullin = tf.constant(np.arange(47*30*400).reshape(47,30,400), dtype=tf.float32)
            curins = tf.unstack(fullin, axis=1)

            readouts = [tf.matmul(curin, WW) for curin in curins]

            predictions = tf.reshape(tf.stack(readouts, axis=1), shape=(47, 30))
            preds = predictions.eval()
            print preds
            print np.mean(preds)

        with tf.Session() as sess:
            WW = tf.constant(np.arange(400*1).reshape(400,1), dtype=tf.float32)

            fullin = tf.constant(np.arange(47*30*400).reshape(47,30,400), dtype=tf.float32)
            #curins = tf.unstack(fullin, axis=1)
            #print len(curins)
            dec_out_matrix = tf.reshape(fullin, shape=(-1, 400))

            readouts = tf.matmul(dec_out_matrix, WW)
            #print readouts.get_shape()

            predictions = tf.reshape(readouts, shape=(47, 30))
            preds2 = predictions.eval()
            print preds2
            print np.mean(preds2)

        assert np.all(preds==preds2)
        """

        # momentum = 0.5
        # tf.reset_default_graph() #kind of redundant statement
        if verbose:
            # print "lamda2: %f" % lamda2
            print "learning rate: %f" % learningRate

        output_seq_len = target_len + self.EOS_TOKEN_LEN

        graph = tf.Graph()  # create new graph

        with graph.as_default():
            with tf.name_scope('data'):
                inputs = tf.placeholder(dtype=self.dtype,
                                        shape=(batch_size, input_len, self.FEATURE_LEN), name="inputs")

                targets = tf.placeholder(dtype=self.dtype, shape=(batch_size, output_seq_len), name="targets")

                # temporary to make it easy for ourselves the first time
                decoder_inputs = tf.placeholder(dtype=self.dtype,
                                                shape=(batch_size, output_seq_len,
                                                       self.TARGET_FEATURE_LEN))

            with tf.name_scope('inputs'):
                # unpack matrix into 1 dim array
                inputs_series = tf.unstack(inputs, axis=1)

                if verbose:
                    print len(inputs_series)
                    print inputs_series[0]  # shape: (5,)
                    print

            with tf.name_scope('encoder_rnn_layer'):
                # init_single_state = tf.get_variable('init_state', [1, num_units],
                #                                     initializer=tf.constant_initializer(0.0, dtype=self.dtype))
                # initial_state = tf.tile(init_single_state, [batch_size, 1])

                # don't really care for encoder outputs, but only for its final state
                # the encoder consumes all the input to get a sense of the trend of price history
                _, encoder_final_state = rnn.static_rnn(cell=rnn_cell(num_units=num_units),
                                                        inputs=inputs_series,
                                                        initial_state=None,
                                                        # TODO when using trunc backprop this should not be zero
                                                        dtype=self.dtype)

                if verbose:
                    print encoder_final_state
                    print

            with tf.variable_scope('decoder_rnn_layer'):
                decoder_inputs_series = tf.unstack(decoder_inputs, axis=1)
                if verbose:
                    print "decoder inputs series"
                    print type(decoder_inputs_series)
                    print len(decoder_inputs_series)
                    print decoder_inputs_series[0]
                    print

                # note that we use the same number of units for decoder here
                decoder_outputs, _ = rnn.static_rnn(cell=rnn_cell(num_units=num_units),
                                                    inputs=decoder_inputs_series,
                                                    initial_state=encoder_final_state,
                                                    dtype=self.dtype)

                if verbose:
                    print len(decoder_outputs)
                    print decoder_outputs[0]
                    print

                    # decoder_logits = tf.contrib.layers.linear(decoder_outputs, target_len)
                    # tf.contrib.layers.fully_connected(decoder_outputs)

            # with tf.name_scope('decoder_outs'):
            #     stacked = tf.stack(decoder_outputs, axis=1)
            #
            #     if verbose:
            #         print stacked
            #         print

            with tf.name_scope('readout_layer'):
                WW = tf.Variable(self.rng.randn(num_units, self.TARGET_FEATURE_LEN), dtype=self.dtype, name='weights')
                bb = tf.Variable(np.zeros(self.TARGET_FEATURE_LEN), dtype=self.dtype, name='bias')
                readouts = [tf.matmul(output, WW) + bb for output in decoder_outputs]

                if verbose:
                    print len(readouts)
                    print readouts[0]
                    print

            with tf.name_scope('predictions'):
                predictions = tf.reshape(tf.stack(readouts, axis=1), shape=(batch_size, output_seq_len))
                if verbose:
                    print predictions
                    print

            with tf.name_scope('error'):
                if cost_func == self.COST_FUNCS.HUBER_LOSS:
                    losses = huberLoss(y_true=targets, y_pred=predictions)  # both have shape: (batch_size, target_len)
                elif cost_func == self.COST_FUNCS.MSE:
                    losses = tfMSE(outputs=predictions, targets=targets)
                else:
                    raise Exception("invalid or non supported cost function")

                if verbose:
                    print losses
                    print

                if self.with_EOS:  # fix error to exclude the EOS from the error calculation
                    mask = np.ones(shape=losses.get_shape())
                    mask[:, -1:] = 0
                    losses = losses * tf.constant(mask, dtype=tf.float32)

                error = tf.reduce_mean(losses)

                if verbose:
                    print error
                    print

            with tf.name_scope('training_step'):
                train_step = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(error)

            init = tf.global_variables_initializer()

        self.init = init
        self.inputs = inputs
        self.targets = targets

        self.error = error
        # self.is_training = is_training
        self.train_step = train_step
        self.predictions = predictions
        self.decoder_inputs = decoder_inputs

        # self.keep_prob_input = keep_prob_input
        # self.keep_prob_hidden = keep_prob_hidden

        return graph
