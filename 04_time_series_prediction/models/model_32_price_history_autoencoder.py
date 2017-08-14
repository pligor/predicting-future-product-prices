from __future__ import division

import numpy as np
import tensorflow as tf
from cost_functions.huber_loss import huber_loss
from data_providers.data_provider_32_price_history_autoencoder import PriceHistoryAutoEncDataProvider
from interfaces.neural_net_model_interface import NeuralNetModelInterface
from mylibs.batch_norm import BatchNormer, batchNormWrapper, fully_connected_layer_with_batch_norm_and_l2, \
    fully_connected_layer_with_batch_norm
from mylibs.jupyter_notebook_helper import DynStats, getRunTime
from tensorflow.contrib import rnn
from collections import OrderedDict
from mylibs.py_helper import merge_dicts
from mylibs.tf_helper import generate_weights_var, fully_connected_layer
from os import system
from fastdtw import fastdtw
from matplotlib import pyplot as plt

from plotter.price_hist import renderRandomMultipleTargetsVsPredictions


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
        self.inputs = None
        self.predictions = None
        self.train_step = None
        self.is_training = None
        self.decoder_extra_inputs = None
        self.keep_prob_rnn_out = None
        self.keep_prob_readout = None
        self.twod = None

    @staticmethod
    def DEFAULT_ACTIVATION_RNN():
        return tf.nn.tanh  # tf.nn.elu

    def run(self, npz_path, epochs, batch_size, enc_num_units, dec_num_units, ts_len,
            #decoder_first_input=DECODER_FIRST_INPUT.ZEROS,
            learning_rate=ADAM_DEFAULT_LEARNING_RATE,
            preds_gather_enabled=True,
            ):

        graph = self.getGraph(batch_size=batch_size, verbose=False, enc_num_units=enc_num_units,
                              dec_num_units=dec_num_units, ts_len=ts_len,
                              learning_rate=learning_rate)
        # input_keep_prob=input_keep_prob, hidden_keep_prob=hidden_keep_prob,

        train_data = PriceHistoryAutoEncDataProvider(npz_path=npz_path, batch_size=batch_size, rng=self.rng,
                                                     which_set='train', ts_max_len=ts_len)
        # during cross validation we execute our experiment multiple times and we get a score at the end
        # so this means that we need to retrain the model one final time in order to output the predictions
        # from this training procedure
        preds_dp = PriceHistoryAutoEncDataProvider(npz_path=npz_path, batch_size=batch_size, rng=self.rng,
                                                   shuffle_order=False,
                                                   which_set='test',
                                                   ts_max_len=ts_len,
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

        with tf.Session(graph=graph, config=self.config) as sess:
            sess.run(self.init)  # sess.run(tf.initialize_all_variables())

            dynStats = DynStats(validation=valid_data is not None)

            for epoch in range(epochs):
                train_error, runTime = getRunTime(
                    lambda:
                    self.trainEpoch(
                        sess=sess,
                        data_provider=train_data,
                        extraFeedDict={
                            self.is_training: True,
                        }
                    )
                )

                if np.isnan(train_error):
                    raise Exception('do something with your learning rate because it is extremely high')

                if valid_data is None:
                    if verbose:
                        # print 'EndEpoch%02d(%.3f secs):err(train)=%.4f,acc(train)=%.2f,err(valid)=%.2f,acc(valid)=%.2f, ' % \
                        #       (epoch + 1, runTime, train_error, train_accuracy, valid_error, valid_accuracy)
                        print 'End Epoch %02d (%.3f secs): err(train) = %.6f' % (
                            epoch + 1, runTime, train_error)

                    dynStats.gatherStats(train_error=train_error)
                else:
                    # if (epoch + 1) % 1 == 0:
                    valid_error = self.validateEpoch(
                        sess=sess,
                        data_provider=valid_data,
                        extraFeedDict={self.is_training: False},
                    )

                    if np.isnan(valid_error):
                        raise Exception('do something with your learning rate because it is extremely high')

                    if verbose:
                        print 'End Epoch %02d (%.3f secs): err(train) = %.6f, err(valid)=%.6f' % (
                            epoch + 1, runTime, train_error, valid_error)

                    dynStats.gatherStats(train_error=train_error, valid_error=valid_error)

            preds_dict, test_error = self.getPredictions(batch_size=batch_size, data_provider=preds_dp,
                                                         sess=sess) if preds_gather_enabled else (None, None)

        if verbose:
            if preds_gather_enabled:
                print "total test error: {}".format(test_error)
            print

        if preds_gather_enabled:
            return dynStats, preds_dict, preds_dp.get_targets_dict()
        else:
            return dynStats

    def train_predict(self, npz_path,
                      num_units, epochs, batch_size, ts_len, rnn_hidden_dim,
                      plotting=False,
                      decoder_first_input=DECODER_FIRST_INPUT.ZEROS,
                      lamda2=DEFAULT_ARR_LAMDA2,
                      keep_prob_rnn_out=DEFAULT_KEEP_PROB,
                      keep_prob_readout=DEFAULT_KEEP_PROB,
                      learning_rate=ADAM_DEFAULT_LEARNING_RATE,
                      verbose=True):
        """WE NEED TO FIX THIS, BEFORE USING. IT HAS NEVER BEEN TESTED. IT IS COPY PASTE FROM ANOTHER MODEL"""

        graph = self.getGraph(batch_size=batch_size, verbose=False, enc_num_units=num_units,
                              dec_num_units=num_units, ts_len=ts_len,
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
                                                   )

        self.__print_hyperparams(learning_rate=learning_rate, epochs=epochs, keep_prob_rnn_out=keep_prob_rnn_out,
                                 keep_prob_readout=keep_prob_readout, lamda2=lamda2, enc_num_units=num_units,
                                 dec_num_units=num_units)

        test_error = None
        preds_dict = None

        with tf.Session(graph=graph, config=self.config) as sess:
            sess.run(self.init)  # sess.run(tf.initialize_all_variables())

            dyn_stats = DynStats()

            for epoch in range(epochs):
                train_error, runTime = getRunTime(
                    lambda:
                    self.trainEpoch(
                        sess=sess,
                        data_provider=train_data,
                        extraFeedDict={
                            self.is_training: True,
                            # self.keep_prob_rnn_out: keep_prob_rnn_out,
                            # self.keep_prob_readout: keep_prob_readout,
                        }
                    )
                )

                if np.isnan(train_error):
                    raise Exception('do something with your learning rate because it is extremely high')

                if verbose:
                    # print 'EndEpoch%02d(%.3f secs):err(train)=%.4f,acc(train)=%.2f,err(valid)=%.2f,acc(valid)=%.2f, ' % \
                    #       (epoch + 1, runTime, train_error, train_accuracy, valid_error, valid_accuracy)
                    print 'End Epoch %02d (%.3f secs): err(train) = %.6f' % (
                        epoch + 1, runTime, train_error)

                dyn_stats.gatherStats(train_error=train_error)

                if verbose:
                    preds_dict, test_error = self.getPredictions(batch_size=batch_size, data_provider=preds_dp,
                                                                 sess=sess)

                    targets = preds_dp.targets #recall inputs and targets are the same for an autoencoder
                    dtw_scores = [fastdtw(targets[ind], preds_dict[ind])[0] for ind in range(len(targets))]
                    print "cur dtw score: {}".format(np.mean(dtw_scores))

                    if plotting:
                        renderRandomMultipleTargetsVsPredictions(targets=targets, inputs=preds_dp.inputs,
                                                                 preds=preds_dict.values())
                        plt.show()

        if verbose:
            print "total test error: {}".format(test_error)
            print

        return dyn_stats, preds_dict, preds_dp.get_targets_dict()

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

                decoder_extra_inputs = tf.placeholder(dtype=self.dtype,
                                                      shape=(batch_size, ts_len, self.DATE_FEATURE_LEN),
                                                      name="decoder_extra_inputs")
                self.decoder_extra_inputs = decoder_extra_inputs

                if verbose:
                    print "targets"
                    print targets
                    print

            with tf.name_scope('inputs'):
                # unpack matrix into 1 dim array
                inputs_series = tf.unstack(inputs, axis=1)

                if verbose:
                    print len(inputs_series)
                    print inputs_series[0]  # shape: (batch_size, 1+6)
                    print

            with tf.name_scope('encoder_rnn_layer'):
                # don't really care for encoder outputs, but only for its final state
                # the encoder consumes all the input to get a sense of the trend of price history
                _, encoder_final_state = rnn.static_rnn(
                    cell=tf.contrib.rnn.GRUCell(num_units=enc_num_units, activation=self.DEFAULT_ACTIVATION_RNN()),
                    # cell=tf.contrib.rnn.GRUCell(num_units=enc_num_units),
                    inputs=inputs_series,
                    initial_state=None,
                    dtype=self.dtype
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

                self.twod = affine_enc_out

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

            with tf.name_scope('dec_extra_ins'):
                # unpack matrix
                dec_extra_inputs_series = tf.unstack(decoder_extra_inputs, axis=1)

                if verbose:
                    print len(dec_extra_inputs_series)
                    print dec_extra_inputs_series[0]  # shape: (batch_size, 6) #only date info for the time being
                    print

            with tf.variable_scope('decoder_rnn_layer'):
                decoder_outputs, decoder_final_state = rnn.static_rnn(
                    # cell=tf.contrib.rnn.GRUCell(num_units=dec_num_units, activation=self.DEFAULT_ACTIVATION_RNN),
                    cell=tf.contrib.rnn.GRUCell(num_units=dec_num_units, activation=self.DEFAULT_ACTIVATION_RNN()),
                    inputs=dec_extra_inputs_series,
                    initial_state=dec_init_state,
                    dtype=self.dtype
                )

                if verbose:
                    print "decoder_outputs len: {}".format(len(decoder_outputs))
                    print decoder_outputs[0]
                    print

            with tf.name_scope('decoder_outs'):
                stacked_dec_outs = tf.stack(decoder_outputs, axis=1)
                flattened_dec_outs = tf.reshape(stacked_dec_outs, shape=(-1, dec_num_units))

                if verbose:
                    print stacked_dec_outs
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

                if verbose:
                    print losses
                    print

                loss = tf.reduce_mean(losses)

                error = loss

                if verbose:
                    print loss
                    print error
                    print

            with tf.name_scope('training_step'):
                train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

            init = tf.global_variables_initializer()

        self.init = init
        self.inputs = inputs
        self.error = error
        self.train_step = train_step
        self.predictions = outputs

        return graph

    def getPredictions(self, sess, data_provider, batch_size, extraFeedDict=None):
        if extraFeedDict is None:
            extraFeedDict = {}

        assert data_provider.data_len % batch_size == 0  # provider can support and intermediate values

        total_error = 0.

        instances_order = data_provider.current_order

        target_len = data_provider.targets.shape[1]

        all_predictions = np.zeros(shape=(data_provider.data_len, target_len))

        for inst_ind, (input_batch, dec_extra_ins) in enumerate(data_provider):
            cur_error, cur_preds = sess.run(
                [self.error, self.predictions],
                feed_dict=merge_dicts({self.inputs: input_batch,
                                       self.decoder_extra_inputs: dec_extra_ins,
                                       self.is_training: False,
                                       }, extraFeedDict))

            assert np.all(instances_order == data_provider.current_order)

            all_predictions[inst_ind * batch_size: (inst_ind + 1) * batch_size, :] = cur_preds

            total_error += cur_error

        total_error /= data_provider.num_batches

        if np.any(all_predictions == 0):
            print "all predictions are expected to be something else than absolute zero".upper()
            system('play --no-show-progress --null --channels 1 synth {} sine {}'.format(0.5, 800))
        # assert np.all(all_predictions != 0), "all predictions are expected to be something else than absolute zero"

        preds_dict = OrderedDict(zip(instances_order, all_predictions))

        return preds_dict, total_error

    def validateEpoch(self, sess, data_provider, extraFeedDict=None):
        if extraFeedDict is None:
            extraFeedDict = {}

        total_error = 0.

        num_batches = data_provider.num_batches

        for step, (input_batch, dec_extra_ins) in enumerate(data_provider):
            feed_dic = merge_dicts({self.inputs: input_batch,
                                    self.decoder_extra_inputs: dec_extra_ins,
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

        for step, (input_batch, dec_extra_ins) in enumerate(data_provider):
            feed_dic = merge_dicts({self.inputs: input_batch,
                                    self.decoder_extra_inputs: dec_extra_ins,
                                    }, extraFeedDict)

            _, batch_error = sess.run([self.train_step, self.error], feed_dict=feed_dic)

            train_error += batch_error

        train_error /= num_batches

        return train_error

    @staticmethod
    def __print_hyperparams(**kwargs):
        for key in kwargs:
            print "{}: {}".format(key, kwargs[key])
