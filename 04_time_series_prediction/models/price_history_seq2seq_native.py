from __future__ import division

import numpy as np
import tensorflow as tf

from cost_functions.huber_loss import huberLoss
from data_providers.price_history_seq2seq_data_provider import PriceHistorySeq2SeqDataProvider
from mylibs.jupyter_notebook_helper import DynStats, getRunTime
# from model_selection.cross_validator import CrossValidator
from interfaces.neural_net_model_interface import NeuralNetModelInterface
from tensorflow.contrib import rnn
from collections import OrderedDict
from mylibs.py_helper import merge_dicts
from mylibs.tf_helper import tfMSE
import tensorflow.contrib.seq2seq as seq2seq


class PriceHistorySeq2SeqNative(NeuralNetModelInterface):
    FEATURE_LEN = 1
    TARGET_FEATURE_LEN = 1

    def __init__(self, rng, dtype, config, debug=False, with_EOS=True):
        # super(ShiftFunc, self).__init__(random_state=rng, data_provider_class=ProductPairsDataProvider)
        super(PriceHistorySeq2SeqNative, self).__init__()

        # with bidirectional encoder, decoder state size should be
        # 2x encoder state size

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
        self.debug = debug

        self.EOS_TOKEN_LEN = 1 if with_EOS else 0
        self.with_EOS = with_EOS

    class COST_FUNCS(object):
        HUBER_LOSS = "huberloss"
        MSE = 'mse'

    class RNN_CELLS(object):
        BASIC_RNN = tf.contrib.rnn.BasicRNNCell  # "basic_rnn"

        GRU = tf.contrib.rnn.GRUCell  # "gru"

    def run(self, npz_path, epochs, batch_size, num_units, input_len, target_len,
            eos_token=PriceHistorySeq2SeqDataProvider.EOS_TOKEN_DEFAULT,
            preds_gather_enabled=True,
            cost_func=COST_FUNCS.HUBER_LOSS, rnn_cell=RNN_CELLS.BASIC_RNN):

        graph = self.getGraph(batch_size=batch_size, verbose=False, num_units=num_units, input_len=input_len,
                              rnn_cell=rnn_cell, target_len=target_len, cost_func=cost_func, eos_token=eos_token)

        train_data = PriceHistorySeq2SeqDataProvider(npz_path=npz_path, batch_size=batch_size, rng=self.rng,
                                                     eos_token=eos_token, with_EOS=self.with_EOS)

        preds_dp = PriceHistorySeq2SeqDataProvider(npz_path=npz_path, batch_size=batch_size, rng=self.rng,
                                                   shuffle_order=False, eos_token=eos_token, with_EOS=self.with_EOS,
                                                   ) if preds_gather_enabled else None

        return self.train_validate(train_data=train_data, valid_data=None, graph=graph, epochs=epochs,
                                   preds_gather_enabled=preds_gather_enabled, preds_dp=preds_dp, batch_size=batch_size)

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

        for inst_ind, (input_batch, target_batch) in enumerate(data_provider):
            cur_error, cur_preds = sess.run(
                [self.error, self.predictions],
                feed_dict=merge_dicts({self.inputs: input_batch,
                                       self.targets: target_batch,
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

        for step, (input_batch, target_batch) in enumerate(data_provider):
            feed_dic = merge_dicts({self.inputs: input_batch,
                                    self.targets: target_batch,
                                    }, extraFeedDict)

            _, batch_error = sess.run([self.train_step, self.error], feed_dict=feed_dic)

            train_error += batch_error

        train_error /= num_batches

        return train_error

    def getGraph(self,
                 # TODO in this version we are building it full length and then we are going to improve it (trunc backprop len)
                 batch_size,
                 num_units,
                 input_len,
                 target_len,
                 eos_token=PriceHistorySeq2SeqDataProvider.EOS_TOKEN_DEFAULT,
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
                if self.debug:
                    inputs, targets = self.get_debug_data(batch_size=batch_size, input_len=input_len,
                                                          output_seq_len=output_seq_len)
                else:
                    inputs = tf.placeholder(dtype=self.dtype,
                                            shape=(batch_size, input_len, self.FEATURE_LEN), name="inputs")

                    targets = tf.placeholder(dtype=self.dtype, shape=(batch_size, output_seq_len), name="targets")

            with tf.name_scope('inputs'):
                # unpack matrix into 1 dim array
                inputs_series = tf.unstack(inputs, axis=1)

                if verbose:
                    print len(inputs_series)
                    print inputs_series[0]  # shape: (batch_size, feature_len), here (47,1)
                    print

            with tf.name_scope('encoder_rnn_layer'):
                # with tf.variable_scope("Encoder") as scope:
                #     (self.encoder_outputs, self.encoder_state) = (
                #         tf.nn.dynamic_rnn(cell=self.encoder_cell,
                #                           inputs=self.encoder_inputs_embedded,
                #                           sequence_length=self.encoder_inputs_length,
                #                           time_major=True,
                #                           dtype=tf.float32)
                #     )

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
                # Potentially useful classes / functions from seq2seq module of tf r.1.1 onwards
                # class BasicDecoder: Basic sampling decoder.
                # class BasicDecoderOutput
                # class Decoder: An RNN Decoder abstract interface object.
                # class Helper: Interface for implementing sampling in seq2seq decoders. <--- this is only interface, no implementation
                # dynamic_decode(...): Perform dynamic decoding with decoder.
                # Training Helper: A helper for use during training. Only reads inputs. <-- this requires inputs

                # helper = seq2seq.TrainingHelper()
                # #TODO probably good to use it with the dummy version of the model where we provide the decoder inputs

                # EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.EOS
                eos_token_tensor = tf.constant(np.ones(shape=(batch_size, 1)) * eos_token,
                                               dtype=tf.float32, name='eos_token_tensor')

                WW = tf.Variable(self.rng.randn(num_units, self.TARGET_FEATURE_LEN), dtype=self.dtype, name='weights')
                bb = tf.Variable(np.zeros(self.TARGET_FEATURE_LEN), dtype=self.dtype, name='bias')

                # if we are not working with end of sequence one idea is to use as initial input to the decoder the
                # last input of the inputs
                initial_inputs = eos_token_tensor if self.with_EOS else inputs_series[-1]

                helper = seq2seq.CustomHelper(
                    initialize_fn=self.get_initialize_fn(batch_size=batch_size, initial_inputs=initial_inputs),
                    sample_fn=self.get_sample_fn(batch_size=batch_size),
                    next_inputs_fn=self.get_next_inputs_fn(batch_size=batch_size, static_seq_len=output_seq_len, WW=WW,
                                                           bb=bb)
                )

                decoder = seq2seq.BasicDecoder(cell=rnn_cell(num_units=num_units),
                                               helper=helper,
                                               initial_state=encoder_final_state)

                # https://stackoverflow.com/questions/44483159/how-to-use-tensorflow-v1-1-seq2seq-dynamic-decode
                dyn_decode_outs = seq2seq.dynamic_decode(decoder=decoder,
                                                         output_time_major=False,
                                                         # we have batches as our major
                                                         impute_finished=False,
                                                         # we don't really care because it affects dynamic sequences
                                                         )
                basic_decoder_output, _ = dyn_decode_outs
                if verbose:
                    print basic_decoder_output
                    print

                decoder_out_tensor = basic_decoder_output.rnn_output

                # del final_state, final_sequence_lengths #not interesting <-- doc has it wrong, there is no final_sequence_lengths output
                if verbose:
                    print decoder_out_tensor
                    print

            with tf.name_scope('readout_layer'):
                # just grab the dimensions
                # decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_out_tensor))
                decoder_dim = decoder_out_tensor.get_shape()[-1].value  # last dimension's value

                decoder_outputs_flat = tf.reshape(decoder_out_tensor, (-1, decoder_dim))

                # TODO note that we are using exactly the same that we used above inside RNN cell (not sure if this is the best)
                readouts = tf.add(tf.matmul(decoder_outputs_flat, WW), bb, "readouts")

                if verbose:
                    print readouts
                    print

            with tf.name_scope('predictions'):
                # in purpose batch size goes last and output_seq_len goes second (we omit the target feature len because it is 1)
                # because raw rnn stack things on top of each other and puts sequence length first
                predictions = tf.reshape(readouts, shape=(batch_size, output_seq_len))
                # decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))
                # https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/2-seq2seq-advanced.ipynb
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
                #TODO here we have removed the last element that we pass to the optimizer below, maybe this is wrong,
                #perhaps the optimizer needs all the error even if we only care to report the error without the EOS
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

        # self.keep_prob_input = keep_prob_input
        # self.keep_prob_hidden = keep_prob_hidden

        return graph

    def get_initialize_fn(self, batch_size, initial_inputs):
        def initialize_fn():
            """callable that returns (finished, next_inputs) for the first iteration."""
            finished = self.all_elems_non_finished(batch_size=batch_size)

            next_inputs = initial_inputs

            return (finished,
                    next_inputs)

        return initialize_fn

    def get_sample_fn(self, batch_size):
        def sample_fn(time, outputs, state):
            """ callable that takes (time, outputs, state) and emits tensor sample_ids."""
            # in translation case you have the output being reduced to a number of logits which can be many if the
            # vocabulary size is large and from this logits only one of them is the maximum, so we could consider this
            # as the sample id. But here we do not have such case therefore we will use an invalid id index: -1
            return -1 * tf.ones(shape=(batch_size,),
                                dtype=self.dtype)  # we actually do not care about sampling here, we are not doing classification

        return sample_fn

    def get_next_inputs_fn(self, batch_size, static_seq_len, WW, bb):
        def next_inputs_fn(time, outputs, state, sample_ids):
            """callable that takes (time, outputs, state, sample_ids) and emits (finished, next_inputs, next_state)"""
            finished = self.get_seq_len_tensor(batch_size=batch_size, static_seq_len=static_seq_len) <= time + 1
            next_inputs = tf.add(tf.matmul(outputs, WW), bb)
            next_state = state

            return (finished, next_inputs, next_state)

        return next_inputs_fn

    @staticmethod
    def get_seq_len_tensor(batch_size, static_seq_len):
        return tf.constant(np.full(shape=(batch_size,), fill_value=static_seq_len), dtype=tf.int32)

    @staticmethod
    def all_elems_non_finished(batch_size, finished=False):
        # print "finished"
        # print finished

        # return tf.constant(np.repeat(finished, batch_size), dtype=tf.bool)  # (0 >= decoder_lengths)
        return tf.constant(np.full(shape=(batch_size,), fill_value=finished, dtype=np.bool),
                           dtype=tf.bool)  # (0 >= decoder_lengths)

    def get_debug_data(self, batch_size, input_len, output_seq_len):

        inputs = tf.constant(value=np.arange(batch_size * input_len * self.FEATURE_LEN).reshape(
            shape=(batch_size, input_len, self.FEATURE_LEN)), dtype=self.dtype, name="inputs")

        targets = tf.constant(value=np.arange(batch_size * output_seq_len).reshape(shape=(batch_size, output_seq_len)),
                              dtype=self.dtype, name="targets")

        return inputs, targets
