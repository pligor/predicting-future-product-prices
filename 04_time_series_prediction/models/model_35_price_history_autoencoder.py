from __future__ import division

import numpy as np
import tensorflow as tf
from cost_functions.huber_loss import huber_loss
from data_providers.data_provider_33_price_history_autoencoder import PriceHistoryAutoEncDataProvider
from interfaces.neural_net_model_interface import NeuralNetModelInterface
from mylibs.batch_norm import BatchNormer, batchNormWrapper, fully_connected_layer_with_batch_norm_and_l2, \
    fully_connected_layer_with_batch_norm
from mylibs.jupyter_notebook_helper import DynStats, getRunTime
from tensorflow.contrib import rnn
from collections import OrderedDict
from mylibs.py_helper import merge_dicts
from mylibs.tf_helper import generate_weights_var, fully_connected_layer, tf_diff_axis_1
from os import system
from fastdtw import fastdtw
from matplotlib import pyplot as plt

from plotter.price_hist import renderRandomMultipleTargetsVsPredictions

# DYNAMIC SEQUENCES - PLUS DIFF OPTIMIZER
class PriceHistoryAutoencoder(NeuralNetModelInterface):
    """
    NECESSARY FOR MULTIPLE SEQS:
    - Make it with dynamic inputs

    IDEAS FOR IMPROVEMENT:
    0) introduce extra layers
    1) Add the mobile attributes per instance
    2) MAKE OUTPUT BE DEPENDED ON PREVIOUS OUTPUT
    3) use EOS
    4) Add dropout
    *) Make also input be depende on previous input ??
    """

    DATE_FEATURE_LEN = 6
    INPUT_FEATURE_LEN = DATE_FEATURE_LEN + 1
    TS_INPUT_IND = 0  # if feature len is multi
    TARGET_FEATURE_LEN = 1
    ADAM_DEFAULT_LEARNING_RATE = 1e-3
    SEED = 16011984
    DEFAULT_KEEP_PROB = 1.
    DEFAULT_LAMDA2 = 0.
    DEFAULT_ARR_LAMDA2 = [DEFAULT_LAMDA2] * 3
    BATCH_NORM_ENABLED_BY_DEFAULT = True

    class DECODER_FIRST_INPUT(object):
        PREVIOUS_INPUT = "PREVIOUS_INPUT"
        ZEROS = "ZEROS"

    def __init__(self, rng, dtype, config):
        super(PriceHistoryAutoencoder, self).__init__()
        self.rng = rng
        self.dtype = dtype
        self.config = config
        self.train_data = None
        self.valid_data = None
        self.init = None
        self.error = None
        self.error_diff = None
        self.inputs = None
        self.predictions = None
        self.train_step = None
        self.is_training = None
        self.decoder_extra_inputs = None
        self.keep_prob_rnn_out = None
        self.keep_prob_readout = None
        self.twod = None
        self.sequence_lens = None
        self.sequence_len_mask = None

    @staticmethod
    def DEFAULT_ACTIVATION_RNN():
        return tf.nn.tanh  # tf.nn.elu

    def run(self, npz_path, epochs, batch_size, enc_num_units, dec_num_units, ts_len,
            # decoder_first_input=DECODER_FIRST_INPUT.ZEROS,
            learning_rate=ADAM_DEFAULT_LEARNING_RATE,
            preds_gather_enabled=True,
            ):

        graph = self.getGraph(batch_size=batch_size, verbose=False, enc_num_units=enc_num_units,
                              dec_num_units=dec_num_units, ts_len=ts_len,
                              learning_rate=learning_rate)
        # input_keep_prob=input_keep_prob, hidden_keep_prob=hidden_keep_prob,

        train_data = PriceHistoryAutoEncDataProvider(npz_path=npz_path, batch_size=batch_size, rng=self.rng,
                                                     which_set='train')
        # during cross validation we execute our experiment multiple times and we get a score at the end
        # so this means that we need to retrain the model one final time in order to output the predictions
        # from this training procedure
        preds_dp = PriceHistoryAutoEncDataProvider(npz_path=npz_path, batch_size=batch_size, rng=self.rng,
                                                   shuffle_order=False,
                                                   which_set='test',
                                                   ) if preds_gather_enabled else None

        self.__print_hyperparams(learning_rate=learning_rate, epochs=epochs, enc_num_units=enc_num_units,
                                 dec_num_units=dec_num_units)

        return self.train_validate(train_data=train_data, valid_data=None, graph=graph, epochs=epochs,
                                   preds_gather_enabled=preds_gather_enabled, preds_dp=preds_dp,
                                   batch_size=batch_size)

    def train_validate(self, train_data, valid_data, **kwargs):
        graph = kwargs['graph']
        epochs = kwargs['epochs']
        batch_size = kwargs['batch_size']
        verbose = kwargs['verbose'] if 'verbose' in kwargs.keys() else True
        preds_dp = kwargs['preds_dp'] if 'preds_dp' in kwargs.keys() else None
        preds_gather_enabled = kwargs['preds_gather_enabled'] if 'preds_gather_enabled' in kwargs.keys() else True

        test_error = None
        preds_dict = None
        twod_dict = None
        test_error_diff = None

        with tf.Session(graph=graph, config=self.config) as sess:
            sess.run(self.init)  # sess.run(tf.initialize_all_variables())

            dynStats = DynStats(validation=valid_data is not None)
            dyn_stats_diff = DynStats(validation=valid_data is not None)

            for epoch in range(epochs):
                (train_error, train_error_diff), runTime = getRunTime(
                    lambda:
                    self.trainEpoch(
                        sess=sess,
                        data_provider=train_data,
                        extraFeedDict={
                            self.is_training: True,
                        }
                    )
                )

                if np.isnan(train_error) or np.isnan(train_error_diff):
                    raise Exception('do something with your learning rate because it is extremely high')

                if valid_data is None:
                    if verbose:
                        # print 'EndEpoch%02d(%.3f secs):err(train)=%.4f,acc(train)=%.2f,err(valid)=%.2f,acc(valid)=%.2f, ' % \
                        #       (epoch + 1, runTime, train_error, train_accuracy, valid_error, valid_accuracy)
                        print 'End Epoch %02d (%.3f secs): err(train) = %.6f, err_diff(train) = %.6f' % (
                            epoch + 1, runTime, train_error, train_error_diff)

                    dynStats.gatherStats(train_error=train_error)
                    dyn_stats_diff.gatherStats(train_error=train_error_diff)
                else:
                    # if (epoch + 1) % 1 == 0:
                    valid_error, valid_error_diff = self.validateEpoch(
                        sess=sess,
                        data_provider=valid_data,
                        extraFeedDict={self.is_training: False},
                    )

                    if np.isnan(valid_error) or np.isnan(valid_error_diff):
                        raise Exception('do something with your learning rate because it is extremely high')

                    if verbose:
                        print 'End Epoch%02d(%.3f secs): err(tr)=%.5f, err_diff(tr)=%.5f err(val)=%.5f, err_diff(val)=%.5f' % (
                            epoch + 1, runTime, train_error, train_error_diff, valid_error, valid_error_diff)

                    dynStats.gatherStats(train_error=train_error, valid_error=valid_error)
                    dyn_stats_diff.gatherStats(train_error=train_error_diff, valid_error=valid_error_diff)

            preds_dict, test_error, twod_dict, test_error_diff = self.getPredictions(batch_size=batch_size,
                                                                                      data_provider=preds_dp,
                                                                                      sess=sess) if preds_gather_enabled else (
                None, None, None, None)

        if verbose:
            if preds_gather_enabled:
                print "total test error: {}".format(test_error)
                print "total test diff error: {}".format(test_error_diff)
            print

        dyn_stats_dict = {
            "dyn_stats": dynStats,
            "dyn_stats_diff": dyn_stats_diff
        }

        if preds_gather_enabled:
            return dyn_stats_dict, self.trimPredsDict(preds_dict,
                                                      data_provider=preds_dp), preds_dp.get_targets_dict_trimmed(), twod_dict
        else:
            return dyn_stats_dict

    def getGraph(self,
                 batch_size,
                 enc_num_units,
                 dec_num_units,
                 ts_len,
                 learning_rate=ADAM_DEFAULT_LEARNING_RATE,  # default of Adam is 1e-3
                 verbose=True):
        # momentum = 0.5
        # tf.reset_default_graph() #kind of redundant statement

        graph = tf.Graph()  # create new graph

        with graph.as_default():
            with tf.name_scope('parameters'):
                self.is_training = tf.placeholder(tf.bool, name="is_training")

            with tf.name_scope('data'):
                inputs = tf.placeholder(dtype=self.dtype,
                                        shape=(batch_size, ts_len, self.INPUT_FEATURE_LEN), name="inputs")

                targets = inputs[:, :, self.TS_INPUT_IND]
                if verbose:
                    print "targets"
                    print targets
                    print

                decoder_extra_inputs = tf.placeholder(dtype=self.dtype,
                                                      shape=(batch_size, ts_len, self.DATE_FEATURE_LEN),
                                                      name="decoder_extra_inputs")
                self.decoder_extra_inputs = decoder_extra_inputs

                sequence_lens = tf.placeholder(tf.int32, shape=(batch_size,), name="sequence_lens_placeholder")
                self.sequence_lens = sequence_lens

                sequence_len_mask = tf.placeholder(tf.int32, shape=(batch_size, ts_len),
                                                   name="sequence_len_mask_placeholder")
                self.sequence_len_mask = sequence_len_mask

            with tf.name_scope('encoder_rnn_layer'):
                _, encoder_final_state = tf.nn.dynamic_rnn(
                    cell=tf.contrib.rnn.GRUCell(num_units=enc_num_units, activation=self.DEFAULT_ACTIVATION_RNN()),
                    inputs=inputs,
                    initial_state=None,
                    dtype=self.dtype,
                    sequence_length=sequence_lens
                )

                if verbose:
                    print encoder_final_state
                    print

            with tf.name_scope('encoder_state_out_process'):
                # don't really care for encoder outputs, but only for its final state
                # the encoder consumes all the input to get a sense of the trend of price history

                # fully_connected_layer_with_batch_norm_and_l2(fcId='encoder_state_out_process',
                #                                              inputs=encoder_final_state,
                #                                              input_dim=enc_num_units, output_dim=2,
                #                                              is_training=self.is_training, lamda2=0)
                ww_enc_out = generate_weights_var(ww_id='encoder_state_out_process', input_dim=enc_num_units,
                                                  output_dim=2,
                                                  dtype=self.dtype)
                nonlinearity = tf.nn.elu
                avoidDeadNeurons = 0.1 if nonlinearity == tf.nn.relu else 0.  # prevent zero when relu
                bb_enc_out = tf.Variable(avoidDeadNeurons * tf.ones([2]),
                                         name='biases_{}'.format('encoder_state_out_process'))

                # out_affine = tf.matmul(inputs, weights) + biases
                affine_enc_out = tf.add(tf.matmul(encoder_final_state, ww_enc_out), bb_enc_out)

                self.twod = affine_enc_out  ######### HERE WE GET THE TWO DIM REPRESENTATION OF OUR TIMESERIES ##########

                batchNorm = batchNormWrapper('encoder_state_out_process', affine_enc_out, self.is_training)

                nonlinear_enc_out = nonlinearity(batchNorm)

                if verbose:
                    print nonlinear_enc_out
                    print

            with tf.name_scope('decoder_state_in_process'):
                dec_init_state = fully_connected_layer_with_batch_norm(fcId='decoder_state_in_process',
                                                                       inputs=nonlinear_enc_out,
                                                                       input_dim=2, output_dim=dec_num_units,
                                                                       is_training=self.is_training,
                                                                       nonlinearity=tf.nn.elu)
                if verbose:
                    print dec_init_state
                    print

            with tf.variable_scope('decoder_rnn_layer'):
                decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
                    cell=tf.contrib.rnn.GRUCell(num_units=dec_num_units, activation=self.DEFAULT_ACTIVATION_RNN()),
                    inputs=decoder_extra_inputs,
                    initial_state=dec_init_state,
                    dtype=self.dtype,
                    sequence_length=sequence_lens
                )

                if verbose:
                    print decoder_outputs
                    print

            # No gathering because in this situation we want to keep the entire sequence
            # along with whatever the dynamic_rnn pads at the end

            with tf.name_scope('decoder_outs'):
                flattened_dec_outs = tf.reshape(decoder_outputs, shape=(-1, dec_num_units))

                if verbose:
                    print flattened_dec_outs
                    print

            with tf.name_scope('readout_affine'):
                processed_dec_outs = fully_connected_layer(inputs=flattened_dec_outs,
                                                           input_dim=dec_num_units,
                                                           output_dim=self.TARGET_FEATURE_LEN,
                                                           nonlinearity=tf.identity)

                outputs = tf.reshape(processed_dec_outs, shape=(batch_size, ts_len))

                if verbose:
                    print processed_dec_outs
                    print outputs
                    print

            with tf.name_scope('error'):
                losses = huber_loss(y_true=targets, y_pred=outputs)  # both have shape: (batch_size, target_len)

                # CONVENTION: loss is the error/loss seen by the optimizer, while error is the error reported
                # in the outside world (to us) and usually these two are the same

                lossed_fixed = losses * tf.cast(sequence_len_mask, tf.float32)

                if verbose:
                    print lossed_fixed
                    print

                loss = tf.reduce_mean(lossed_fixed)

                error = loss

                if verbose:
                    print loss
                    print error
                    print

            with tf.name_scope('error_diff'):
                targets_diff = tf_diff_axis_1(targets)
                outputs_diff = tf_diff_axis_1(outputs)

                # because of diff we are losing the first elements and thus the sequence mask becomes
                # just a little bit smaller by dropping the first column
                seq_mask_without_first_elem = sequence_len_mask[:, 1:]
                losses_diff = huber_loss(y_true=targets_diff, y_pred=outputs_diff) * tf.cast(
                    seq_mask_without_first_elem,
                    tf.float32)

                if verbose:
                    print losses_diff
                    print

                loss_diff = tf.reduce_mean(losses_diff)

                error_diff = loss_diff

                if verbose:
                    print loss_diff
                    print error_diff
                    print

            with tf.name_scope('training_step'):
                # TODO we might need to use two different learning rates here
                train_step_amplitude = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
                train_step_diff = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_diff)
                train_step = tf.group(*[train_step_amplitude, train_step_diff])

            init = tf.global_variables_initializer()

        self.init = init
        self.inputs = inputs

        self.error = error
        self.error_diff = error_diff

        self.train_step = train_step
        self.predictions = outputs

        return graph

    def trimPredsDict(self, preds_dict, data_provider):
        assert np.all(np.array(list(data_provider.current_order)) == np.array(list(preds_dict.keys())))

        preds_dict_trimmed = OrderedDict()

        for seqlen, (key, preds) in zip(data_provider.seqlens, preds_dict.iteritems()):
            preds_dict_trimmed[key] = preds[:seqlen]

        return preds_dict_trimmed

    def getPredictions(self, sess, data_provider, batch_size, extraFeedDict=None):
        if extraFeedDict is None:
            extraFeedDict = {}

        assert data_provider.data_len % batch_size == 0  # provider can support and intermediate values

        total_error = 0.
        total_error_diff = 0.

        instances_order = data_provider.current_order

        target_len = data_provider.targets.shape[1]

        all_predictions = np.zeros(shape=(data_provider.data_len, target_len))
        all_two_dims = np.zeros(shape=(data_provider.data_len, 2))

        for inst_ind, (input_batch, dec_extra_ins, seq_lens, seq_len_mask) in enumerate(data_provider):
            cur_error, cur_preds, cur_twod, cur_error_diff = sess.run(
                [self.error, self.predictions, self.twod, self.error_diff],
                feed_dict=merge_dicts({self.inputs: input_batch,
                                       self.decoder_extra_inputs: dec_extra_ins,
                                       self.sequence_lens: seq_lens,
                                       self.sequence_len_mask: seq_len_mask,
                                       self.is_training: False,
                                       }, extraFeedDict))

            assert np.all(instances_order == data_provider.current_order), \
                "making sure that the order does not change as we iterate over our batches"

            cur_batch_slice = slice(inst_ind * batch_size, (inst_ind + 1) * batch_size)

            all_predictions[cur_batch_slice, :] = cur_preds

            all_two_dims[cur_batch_slice, :] = cur_twod

            total_error += cur_error
            total_error_diff += cur_error_diff

        total_error /= data_provider.num_batches
        total_error_diff /= data_provider.num_batches

        if np.any(all_predictions == 0):
            print "all predictions are expected to be something else than absolute zero".upper()
            system('play --no-show-progress --null --channels 1 synth {} sine {}'.format(0.5, 800))
        # assert np.all(all_predictions != 0), "all predictions are expected to be something else than absolute zero"

        preds_dict = OrderedDict(zip(instances_order, all_predictions))
        twod_dict = OrderedDict(zip(instances_order, all_two_dims))

        return preds_dict, total_error, twod_dict, total_error_diff

    def validateEpoch(self, sess, data_provider, extraFeedDict=None):
        if extraFeedDict is None:
            extraFeedDict = {}

        total_error = 0.
        total_error_diff = 0.

        num_batches = data_provider.num_batches

        for step, (input_batch, dec_extra_ins, seq_lens, seq_len_mask) in enumerate(data_provider):
            feed_dic = merge_dicts({self.inputs: input_batch,
                                    self.decoder_extra_inputs: dec_extra_ins,
                                    self.sequence_lens: seq_lens,
                                    self.sequence_len_mask: seq_len_mask,
                                    }, extraFeedDict)

            batch_error, batch_error_diff = sess.run([self.error, self.error_diff], feed_dict=feed_dic)

            total_error += batch_error
            total_error_diff += batch_error_diff

        total_error /= num_batches
        total_error_diff /= num_batches

        return total_error, total_error_diff

    def trainEpoch(self, sess, data_provider, extraFeedDict=None):
        if extraFeedDict is None:
            extraFeedDict = {}

        total_error = 0.
        total_error_diff = 0.

        num_batches = data_provider.num_batches

        for step, (input_batch, dec_extra_ins, seq_lens, seq_len_mask) in enumerate(data_provider):
            feed_dic = merge_dicts({self.inputs: input_batch,
                                    self.decoder_extra_inputs: dec_extra_ins,
                                    self.sequence_lens: seq_lens,
                                    self.sequence_len_mask: seq_len_mask,
                                    }, extraFeedDict)

            _, batch_error, batch_error_diff = sess.run([self.train_step, self.error, self.error_diff],
                                                        feed_dict=feed_dic)

            total_error += batch_error
            total_error_diff += batch_error_diff

        total_error /= num_batches
        total_error_diff /= num_batches

        return total_error, total_error_diff

    @staticmethod
    def __print_hyperparams(**kwargs):
        for key in kwargs:
            print "{}: {}".format(key, kwargs[key])
