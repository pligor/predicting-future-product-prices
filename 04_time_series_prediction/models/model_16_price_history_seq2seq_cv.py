from __future__ import division

import numpy as np
import tensorflow as tf

from cost_functions.huber_loss import huberLoss
from data_providers.price_history_seq2seq_data_provider import PriceHistorySeq2SeqDataProvider
from mylibs.jupyter_notebook_helper import DynStats, getRunTime
from tensorflow.contrib import rnn
from collections import OrderedDict
from mylibs.py_helper import merge_dicts
from mylibs.tf_helper import tfMSE
from model_selection.cross_validator import CrossValidator


class PriceHistorySeq2SeqCV(CrossValidator):
    FEATURE_LEN = 1
    TARGET_FEATURE_LEN = 1
    ADAM_DEFAULT_LEARNING_RATE = 1e-3

    def __init__(self, rng, dtype, config, with_EOS=False):
        super(PriceHistorySeq2SeqCV, self).__init__(random_state=rng,
                                                    data_provider_class=PriceHistorySeq2SeqDataProvider,
                                                    stratified=False)

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

        self.EOS_TOKEN_LEN = 1 if with_EOS else 0
        self.with_EOS = with_EOS

    class COST_FUNCS(object):
        HUBER_LOSS = "huberloss"
        MSE = 'mse'

    class RNN_CELLS(object):
        BASIC_RNN = tf.contrib.rnn.BasicRNNCell  # "basic_rnn"

        GRU = tf.contrib.rnn.GRUCell  # "gru"

    def get_cross_validation_score(self, npz_path, epochs, batch_size, num_units, input_len, target_len, n_splits,
                                   learning_rate=ADAM_DEFAULT_LEARNING_RATE,
                                   eos_token=PriceHistorySeq2SeqDataProvider.EOS_TOKEN_DEFAULT,
                                   cost_func=COST_FUNCS.HUBER_LOSS, rnn_cell=RNN_CELLS.GRU):
        # TODO what we are keeping from the entire cross validation process to be our CV score it is up to us (or isn't?)
        """For CV score we are going to keep the mean value of the minimum validation errors"""

        graph = self.getGraph(batch_size=batch_size, verbose=False, num_units=num_units, input_len=input_len,
                              rnn_cell=rnn_cell, target_len=target_len, cost_func=cost_func, eos_token=eos_token,
                              learning_rate=learning_rate)

        stats_list = self.cross_validate(n_splits=n_splits,
                                         batch_size=batch_size,
                                         graph=graph,
                                         epochs=epochs,
                                         preds_gather_enabled=False,
                                         data_provider_params={
                                             "rng": self.rng,
                                             "batch_size": batch_size,
                                             "npz_path": npz_path,
                                             'eos_token': eos_token,
                                             'with_EOS': self.with_EOS,
                                         })
        assert len(stats_list) > 0, "the stats list should not be empty, a nothing experiment does not make sense"
        #return stats_list
        valid_err_ind = stats_list[0].keys['error(valid)']

        min_valid_errs = [np.min(stat.stats[:, valid_err_ind]) for stat in stats_list]

        cv_score = np.mean(min_valid_errs)

        return cv_score, stats_list

    def run(self, npz_path, epochs, batch_size, num_units, input_len, target_len,
            learning_rate=ADAM_DEFAULT_LEARNING_RATE,
            eos_token=PriceHistorySeq2SeqDataProvider.EOS_TOKEN_DEFAULT,
            preds_gather_enabled=True,
            cost_func=COST_FUNCS.HUBER_LOSS, rnn_cell=RNN_CELLS.GRU):

        graph = self.getGraph(batch_size=batch_size, verbose=False, num_units=num_units, input_len=input_len,
                              rnn_cell=rnn_cell, target_len=target_len, cost_func=cost_func, eos_token=eos_token,
                              learning_rate=learning_rate)
        # input_keep_prob=input_keep_prob, hidden_keep_prob=hidden_keep_prob,

        ############For Predictions Only#################
        train_data = PriceHistorySeq2SeqDataProvider(npz_path=npz_path, batch_size=batch_size, rng=self.rng,
                                                     eos_token=eos_token, with_EOS=self.with_EOS)
        # during cross validation we execute our experiment multiple times and we get a score at the end
        # so this means that we need to retrain the model one final time in order to output the predictions
        # from this training procedure
        preds_dp = PriceHistorySeq2SeqDataProvider(npz_path=npz_path, batch_size=batch_size, rng=self.rng,
                                                   shuffle_order=False, eos_token=eos_token, with_EOS=self.with_EOS,
                                                   ) if preds_gather_enabled else None

        self.__print_hyperparams(learning_rate=learning_rate, epochs=epochs, num_units=num_units)

        stats, preds_dict = self.train_validate(train_data=train_data, valid_data=None, graph=graph, epochs=epochs,
                                                preds_gather_enabled=preds_gather_enabled, preds_dp=preds_dp,
                                                batch_size=batch_size)

        return stats, preds_dict

    @staticmethod
    def __print_hyperparams(learning_rate, epochs, num_units):
        print "learning rate: {:f}".format(learning_rate)
        print "epochs: {:d}".format(epochs)
        print "num units: {:d}".format(num_units)

    def train_validate(self, train_data, valid_data, **kwargs):
        graph = kwargs['graph']
        epochs = kwargs['epochs']
        batch_size = kwargs['batch_size']
        verbose = kwargs['verbose'] if 'verbose' in kwargs.keys() else True
        preds_dp = kwargs['preds_dp'] if 'preds_dp' in kwargs.keys() else None
        preds_gather_enabled = kwargs['preds_gather_enabled'] if 'preds_gather_enabled' in kwargs.keys() else True

        with tf.Session(graph=graph, config=self.config) as sess:
            sess.run(self.init)  # sess.run(tf.initialize_all_variables())

            dynStats = DynStats(validation=valid_data is not None)

            for epoch in range(epochs):
                train_error, runTime = getRunTime(
                    lambda:
                    self.trainEpoch(
                        sess=sess,
                        data_provider=train_data,
                    )
                )
                if valid_data is None:
                    if verbose:
                        # print 'EndEpoch%02d(%.3f secs):err(train)=%.4f,acc(train)=%.2f,err(valid)=%.2f,acc(valid)=%.2f, ' % \
                        #       (epoch + 1, runTime, train_error, train_accuracy, valid_error, valid_accuracy)
                        print 'End Epoch %02d (%.3f secs): err(train) = %.3f' % (epoch + 1, runTime, train_error)

                    dynStats.gatherStats(train_error=train_error)
                else:
                    # if (epoch + 1) % 1 == 0:
                    valid_error = self.validateEpoch(
                        sess=sess,
                        data_provider=valid_data,
                        # extraFeedDict={self.is_training: False},
                        # keep_prob_keys=[self.keep_prob_input, self.keep_prob_hidden]
                    )

                    if verbose:
                        print 'End Epoch %02d (%.3f secs): err(train) = %.3f, err(valid)=%.3f' % (
                            epoch + 1, runTime, train_error, valid_error)

                    dynStats.gatherStats(train_error=train_error, valid_error=valid_error)

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

    def validateEpoch(self, sess, data_provider, extraFeedDict=None):
        if extraFeedDict is None:
            extraFeedDict = {}

        total_error = 0.

        num_batches = data_provider.num_batches

        for step, (input_batch, target_batch) in enumerate(data_provider):
            feed_dic = merge_dicts({self.inputs: input_batch,
                                    self.targets: target_batch,
                                    }, extraFeedDict)

            batch_error = sess.run(self.error, feed_dict=feed_dic)

            total_error += batch_error

        total_error /= num_batches

        return total_error

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
                 rnn_cell=RNN_CELLS.GRU,
                 cost_func=COST_FUNCS.HUBER_LOSS,
                 learning_rate=ADAM_DEFAULT_LEARNING_RATE,  # default of Adam is 1e-3
                 # lamda2=1e-2,
                 verbose=True):
        # momentum = 0.5
        # tf.reset_default_graph() #kind of redundant statement
        output_seq_len = target_len + self.EOS_TOKEN_LEN

        graph = tf.Graph()  # create new graph

        with graph.as_default():
            with tf.name_scope('data'):
                inputs = tf.placeholder(dtype=self.dtype,
                                        shape=(batch_size, input_len, self.FEATURE_LEN), name="inputs")

                targets = tf.placeholder(dtype=self.dtype, shape=(batch_size, output_seq_len), name="targets")

                # temporary to make it easy for ourselves the first time
                # decoder_inputs = tf.placeholder(dtype=self.dtype,
                #                                 shape=(batch_size, output_seq_len,
                #                                        self.TARGET_FEATURE_LEN))

            with tf.name_scope('inputs'):
                # unpack matrix into 1 dim array
                inputs_series = tf.unstack(inputs, axis=1)

                if verbose:
                    print len(inputs_series)
                    print inputs_series[0]  # shape: (5,)
                    print

            with tf.name_scope('encoder_rnn_layer'):
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

            # BAD idea
            # decoder_outputs = []
            # inout = [eos_token]
            # for ii in range(output_seq_len):
            #     if verbose:
            #         print "rnn: {}".format(ii)
            #     with tf.variable_scope('decoder_rnn_layer_{}'.format(ii)):
            #         # note that we use the same number of units for decoder here
            #         inout, _ = rnn.static_rnn(cell=rnn_cell(num_units=num_units),
            #                                   inputs=inout,
            #                                   initial_state=encoder_final_state,
            #                                   dtype=self.dtype)
            #         decoder_outputs += inout

            with tf.variable_scope('decoder_rnn_layer'):
                eos_token_tensor = tf.constant(np.ones(shape=(batch_size, 1)) * eos_token,
                                               dtype=tf.float32, name='eos_token')
                # if we are not working with end of sequence architecture then we use as initial input to the decoder the
                # last value of the inputs
                initial_inputs = eos_token_tensor if self.with_EOS else inputs_series[-1]

                WW = tf.Variable(self.rng.randn(num_units, self.TARGET_FEATURE_LEN), dtype=self.dtype, name='weights')
                bb = tf.Variable(np.zeros(self.TARGET_FEATURE_LEN), dtype=self.dtype, name='bias')

                decoder_output_tensor_array, _, _ = tf.nn.raw_rnn(cell=rnn_cell(num_units=num_units),
                                                                  loop_fn=self.get_loop_fn(
                                                                      encoder_final_state=encoder_final_state,
                                                                      eos_token=initial_inputs,
                                                                      batch_size=batch_size,
                                                                      WW=WW, bb=bb, static_seq_len=output_seq_len,
                                                                  ))
                # del decoder_final_state, decoder_final_loop_state #not interesting

                if verbose:
                    # print len(decoder_outputs)
                    # print decoder_outputs[0]
                    print decoder_output_tensor_array
                    print

                # https://www.tensorflow.org/api_docs/python/tf/TensorArray
                decoder_out_tensor = decoder_output_tensor_array.stack(name='decoder_out_tensor')
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
                predictions_transposed = tf.reshape(readouts, shape=(output_seq_len, batch_size))
                predictions = tf.transpose(predictions_transposed)
                # decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))
                # https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/2-seq2seq-advanced.ipynb
                if verbose:
                    print predictions_transposed
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

                # fix error to exclude the EOS from the error calculation
                if self.with_EOS:  # fix error to exclude the EOS from the error calculation
                    mask = np.ones(shape=losses.get_shape())
                    mask[:, -1:] = 0
                    losses = losses * tf.constant(mask, dtype=tf.float32)

                error = tf.reduce_mean(losses)

                if verbose:
                    print error
                    print

            with tf.name_scope('training_step'):
                train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error)

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

    def get_loop_fn(self, encoder_final_state, eos_token, batch_size, WW, bb, static_seq_len):
        def loop_fn(time, previous_output, previous_state, previous_loop_state):
            # inputs:  time, previous_cell_output, previous_cell_state, previous_loop_state
            # outputs: elements_finished, input, cell_state, output, loop_state

            if previous_state is None:  # time == 0
                assert previous_output is None
                return self.loop_fn_initial(encoder_final_state=encoder_final_state, eos_token=eos_token,
                                            batch_size=batch_size)
            else:
                return self.loop_fn_transition(time=time, previous_cell_output=previous_output,
                                               previous_cell_state=previous_state,
                                               batch_size=batch_size, WW=WW, bb=bb, static_seq_len=static_seq_len)

        return loop_fn

    def loop_fn_initial(self, encoder_final_state, eos_token, batch_size):
        # https://www.tensorflow.org/api_docs/python/tf/nn/raw_rnn
        # here we have a static rnn case so all length so be taken into account,
        # so I guess they are all False always
        # From documentation: a boolean Tensor of shape [batch_size]
        initial_elements_finished = self.all_elems_non_finished(batch_size=batch_size)

        initial_input = eos_token
        initial_cell_state = encoder_final_state

        initial_cell_output = None
        # give it the shape that we want but how exactly ???:
        # initial_cell_output = tf.Variable(np.zeros(shape=(batch_size, self.TARGET_FEATURE_LEN)), dtype=tf.float32)

        initial_loop_state = None  # we don't need to pass any additional information

        return (initial_elements_finished,
                initial_input,
                initial_cell_state,
                initial_cell_output,
                initial_loop_state)

    def loop_fn_transition(self, time, previous_cell_output, previous_cell_state, batch_size, WW, bb, static_seq_len):
        """note that the matrix W is going to be shared among outputs"""
        # print "previous cell output!"
        # print previous_cell_output
        # print

        # print "time: {}".format(time)
        # print

        # finished = self.all_elems_finished(batch_size=batch_size,
        #                                    finished=time - 1 >= static_seq_len)  # (time >= decoder_lengths)
        # finished = self.all_elems_finished(batch_size=batch_size,
        #                                    finished=time - 1 >= static_seq_len)  # (time >= decoder_lengths)
        finished = time >= self.get_seq_len_tensor(batch_size=batch_size, static_seq_len=static_seq_len)

        # this operation produces boolean tensor of [batch_size] defining if corresponding sequence has ended

        # this is always false in our case so just comment next two lines
        # finished = tf.reduce_all(elements_finished)  # -> boolean scalar
        # input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
        next_input = tf.add(tf.matmul(previous_cell_output, WW), bb)

        # print "next input!"
        # print next_input
        # print

        next_cell_state = previous_cell_state
        # emit_output = tf.identity(next_input),
        emit_output = previous_cell_output

        next_loop_state = None  # we don't need to pass any additional information

        return (finished,
                next_input,
                next_cell_state,
                emit_output,
                next_loop_state)

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
