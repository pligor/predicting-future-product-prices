from __future__ import division

import numpy as np
import tensorflow as tf
from cost_functions.huber_loss import huber_loss
from data_providers.data_provider_27_price_history_mobattrs_date import PriceHistoryMobAttrsDateDataProvider
from mylibs.batch_norm import BatchNormer, batchNormWrapper, fully_connected_layer_with_batch_norm_and_l2
from mylibs.jupyter_notebook_helper import DynStats, getRunTime
from tensorflow.contrib import rnn
from collections import OrderedDict
from mylibs.py_helper import merge_dicts
from mylibs.tf_helper import tfMSE
from model_selection.cross_validator import CrossValidator
from mylibs.tf_helper import generate_weights_var
from os import system
from fastdtw import fastdtw
from matplotlib import pyplot as plt

from plotter.price_hist import renderRandomMultipleTargetsVsPredictions


class PriceHistoryMobAttrsDateModel(CrossValidator):
    DATE_FEATURE_LEN = 6
    FEATURE_LEN = DATE_FEATURE_LEN + 1
    TARGET_FEATURE_LEN = 1
    ADAM_DEFAULT_LEARNING_RATE = 1e-3
    SEED = 16011984
    DEFAULT_KEEP_PROB = 1.
    DEFAULT_LAMDA2 = 0.
    DEFAULT_ARR_LAMDA2 = [DEFAULT_LAMDA2] * 3
    DECODER_INPUTS_DEFAULT_PERCENTAGE_USAGE = 0.
    BATCH_NORM_ENABLED_BY_DEFAULT = True

    class COST_FUNCS(object):
        HUBER_LOSS = "huberloss"
        MSE = 'mse'

    DEFAULT_COST_FUNC = COST_FUNCS.MSE

    MOBILE_ATTRS_LEN = 139  # let's consider it static since we are going to be using all of them for the beginning

    class DECODER_FIRST_INPUT(object):
        PREVIOUS_INPUT = "PREVIOUS_INPUT"
        ZEROS = "ZEROS"

    def __init__(self, rng, dtype, config, with_EOS=False):
        super(PriceHistoryMobAttrsDateModel, self).__init__(random_state=rng,
                                                            data_provider_class=PriceHistoryMobAttrsDateDataProvider,
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
        self.is_training = None
        self.dec_ins_percent_usage = None
        self.decoder_inputs = None
        # self.keep_prob_input = None
        self.mobile_attrs = None
        self.decoder_extra_inputs = None
        self.EOS_TOKEN_LEN = 1 if with_EOS else 0
        self.with_EOS = with_EOS
        self.keep_prob_rnn_out = None
        self.keep_prob_readout = None

    DEFAULT_ACTIVATION_RNN = tf.nn.tanh  # tf.nn.elu

    class RNN_CELLS(object):

        # BASIC_RNN = tf.contrib.rnn.BasicRNNCell  # "basic_rnn"
        # @staticmethod
        # def LSTM(num_units):
        #     return tf.contrib.rnn.BasicLSTMCell(num_units=num_units, state_is_tuple=True)

        @staticmethod
        def GRU(activation):
            def get_cell(num_units):
                return tf.contrib.rnn.GRUCell(num_units=num_units, activation=activation)

            return get_cell

    def get_cross_validation_score(self, npz_path, epochs, batch_size, input_len, target_len, n_splits,
                                   enc_num_units,  # TEST THOSE ten hyperparams!!
                                   dec_num_units,
                                   rnn_hidden_dim,
                                   mobile_attrs_dim,
                                   lamda2=DEFAULT_ARR_LAMDA2,
                                   keep_prob_rnn_out=DEFAULT_KEEP_PROB,
                                   keep_prob_readout=DEFAULT_KEEP_PROB,
                                   learning_rate=ADAM_DEFAULT_LEARNING_RATE,

                                   # DO NOT TEST
                                   activation_rnn=DEFAULT_ACTIVATION_RNN,
                                   rnn_cell=RNN_CELLS.GRU,
                                   decoder_first_input=DECODER_FIRST_INPUT.ZEROS,
                                   batch_norm_enabled=BATCH_NORM_ENABLED_BY_DEFAULT,
                                   eos_token=PriceHistoryMobAttrsDateDataProvider.EOS_TOKEN_DEFAULT,
                                   cost_func=DEFAULT_COST_FUNC,
                                   ):
        # what we are keeping from the entire cross validation process to be our CV score it is up to us (or isn't?)
        """For CV score we are going to keep the mean value of the minimum validation errors"""

        graph = self.getGraph(batch_size=batch_size, verbose=False, enc_num_units=enc_num_units,
                              dec_num_units=dec_num_units,
                              input_len=input_len,
                              rnn_cell=rnn_cell(activation_rnn), target_len=target_len, cost_func=cost_func,
                              eos_token=eos_token,
                              learning_rate=learning_rate, lamda2=lamda2, batch_norm_enabled=batch_norm_enabled,
                              decoder_first_input=decoder_first_input, rnn_hidden_dim=rnn_hidden_dim,
                              mobile_attrs_dim=mobile_attrs_dim)

        tuples = self.cross_validate(n_splits=n_splits,
                                     batch_size=batch_size,
                                     graph=graph,
                                     epochs=epochs,
                                     preds_gather_enabled=True,
                                     keep_prob_rnn_out=keep_prob_rnn_out,
                                     keep_prob_readout=keep_prob_readout,
                                     data_provider_params={
                                         "rng": self.rng,
                                         "batch_size": batch_size,
                                         "npz_path": npz_path,
                                         'eos_token': eos_token,
                                         'with_EOS': self.with_EOS,
                                     })
        assert len(tuples) > 0, "the stats list should not be empty, a nothing experiment does not make sense"
        # valid_err_ind = stats_list[0].keys['error(valid)']
        # min_valid_errs = [np.min(stat.stats[:, valid_err_ind]) for stat in stats_list]
        # cv_score = np.mean(min_valid_errs)
        stats_list = [stats_preds[0] for stats_preds in tuples]

        dtw_experiment_scores = []
        for _, preds, targets in tuples:
            cur_len = len(preds)
            assert cur_len == len(targets)

            cur_dtw_scores = []

            assert isinstance(preds, OrderedDict) or isinstance(preds, dict)

            for cur_key in preds:
                cur_dtw_score = fastdtw(targets[cur_key], preds[cur_key])[0]
                cur_dtw_scores.append(cur_dtw_score)

            dtw_experiment_score = np.mean(cur_dtw_scores)
            dtw_experiment_scores.append(dtw_experiment_score)

        cv_score = np.mean(dtw_experiment_scores)

        print "cv score: {}".format(cv_score)

        return cv_score, stats_list

    def run(self, npz_path, epochs, batch_size, enc_num_units, dec_num_units, input_len, target_len, rnn_hidden_dim,
            mobile_attrs_dim,
            decoder_first_input=DECODER_FIRST_INPUT.ZEROS,
            activation_rnn=DEFAULT_ACTIVATION_RNN,
            batch_norm_enabled=BATCH_NORM_ENABLED_BY_DEFAULT,
            lamda2=DEFAULT_ARR_LAMDA2,
            keep_prob_rnn_out=DEFAULT_KEEP_PROB,
            keep_prob_readout=DEFAULT_KEEP_PROB,
            learning_rate=ADAM_DEFAULT_LEARNING_RATE,
            eos_token=PriceHistoryMobAttrsDateDataProvider.EOS_TOKEN_DEFAULT,
            preds_gather_enabled=True,
            cost_func=DEFAULT_COST_FUNC,
            rnn_cell=RNN_CELLS.GRU,
            ):

        graph = self.getGraph(batch_size=batch_size, verbose=False, enc_num_units=enc_num_units,
                              dec_num_units=dec_num_units, input_len=input_len, rnn_cell=rnn_cell(activation_rnn),
                              target_len=target_len, cost_func=cost_func, eos_token=eos_token,
                              learning_rate=learning_rate, lamda2=lamda2, batch_norm_enabled=batch_norm_enabled,
                              decoder_first_input=decoder_first_input, rnn_hidden_dim=rnn_hidden_dim,
                              mobile_attrs_dim=mobile_attrs_dim)
        # input_keep_prob=input_keep_prob, hidden_keep_prob=hidden_keep_prob,

        train_data = PriceHistoryMobAttrsDateDataProvider(npz_path=npz_path, batch_size=batch_size, rng=self.rng,
                                                          eos_token=eos_token, with_EOS=self.with_EOS,
                                                          which_set='train')
        # during cross validation we execute our experiment multiple times and we get a score at the end
        # so this means that we need to retrain the model one final time in order to output the predictions
        # from this training procedure
        preds_dp = PriceHistoryMobAttrsDateDataProvider(npz_path=npz_path, batch_size=batch_size, rng=self.rng,
                                                        shuffle_order=False, eos_token=eos_token,
                                                        with_EOS=self.with_EOS,
                                                        which_set='test',
                                                        ) if preds_gather_enabled else None

        self.__print_hyperparams(learning_rate=learning_rate, epochs=epochs, keep_prob_rnn_out=keep_prob_rnn_out,
                                 keep_prob_readout=keep_prob_readout, lamda2=lamda2, enc_num_units=enc_num_units,
                                 dec_num_units=dec_num_units)

        return self.train_validate(train_data=train_data, valid_data=None, graph=graph, epochs=epochs,
                                   preds_gather_enabled=preds_gather_enabled, preds_dp=preds_dp,
                                   batch_size=batch_size, keep_prob_rnn_out=keep_prob_rnn_out,
                                   keep_prob_readout=keep_prob_readout, )

    def train_validate(self, train_data, valid_data, **kwargs):
        graph = kwargs['graph']
        epochs = kwargs['epochs']
        batch_size = kwargs['batch_size']
        verbose = kwargs['verbose'] if 'verbose' in kwargs.keys() else True
        preds_dp = kwargs['preds_dp'] if 'preds_dp' in kwargs.keys() else None
        preds_gather_enabled = kwargs['preds_gather_enabled'] if 'preds_gather_enabled' in kwargs.keys() else True
        keep_prob_rnn_out = kwargs[
            'keep_prob_rnn_out'] if 'keep_prob_rnn_out' in kwargs.keys() else self.DEFAULT_KEEP_PROB
        keep_prob_readout = kwargs[
            'keep_prob_readout'] if 'keep_prob_readout' in kwargs.keys() else self.DEFAULT_KEEP_PROB

        test_error = None
        preds_dict = None

        with tf.Session(graph=graph, config=self.config) as sess:
            sess.run(self.init)  # sess.run(tf.initialize_all_variables())

            dynStats = DynStats(validation=valid_data is not None)

            # let's use a simple linear progression for now
            first_part = 1  # epochs // 4 <-- very quick pretraining
            dec_ins_percent_usages = np.concatenate((np.linspace(1., 0., first_part), np.zeros(epochs - first_part)))

            for dec_ins_percent_usage, epoch in zip(dec_ins_percent_usages, range(epochs)):
                train_error, runTime = getRunTime(
                    lambda:
                    self.trainEpoch(
                        sess=sess,
                        data_provider=train_data,
                        extraFeedDict={
                            self.is_training: True,
                            self.keep_prob_rnn_out: keep_prob_rnn_out,
                            self.keep_prob_readout: keep_prob_readout,
                            self.dec_ins_percent_usage: dec_ins_percent_usage,
                        }
                    )
                )

                if np.isnan(train_error):
                    raise Exception('do something with your learning rate because it is extremely high')

                if valid_data is None:
                    if verbose:
                        # print 'EndEpoch%02d(%.3f secs):err(train)=%.4f,acc(train)=%.2f,err(valid)=%.2f,acc(valid)=%.2f, ' % \
                        #       (epoch + 1, runTime, train_error, train_accuracy, valid_error, valid_accuracy)
                        print 'End Epoch %02d (%.3f secs): err(train) = %.6f, current dec_ins_percent_usage: %.2f' % (
                            epoch + 1, runTime, train_error, dec_ins_percent_usage)

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
                        print 'End Epoch %02d (%.3f secs): err(train) = %.6f, err(valid)=%.6f, current dec_ins_percent_usage: %.2f' % (
                            epoch + 1, runTime, train_error, valid_error, dec_ins_percent_usage)

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
                      num_units, epochs, batch_size, input_len, target_len, rnn_hidden_dim, mobile_attrs_dim,
                      dec_inputs_part = 1,
                      plotting=False,
                      decoder_first_input=DECODER_FIRST_INPUT.ZEROS,
                      activation_rnn=DEFAULT_ACTIVATION_RNN,
                      batch_norm_enabled=BATCH_NORM_ENABLED_BY_DEFAULT,
                      lamda2=DEFAULT_ARR_LAMDA2,
                      keep_prob_rnn_out=DEFAULT_KEEP_PROB,
                      keep_prob_readout=DEFAULT_KEEP_PROB,
                      learning_rate=ADAM_DEFAULT_LEARNING_RATE,
                      eos_token=PriceHistoryMobAttrsDateDataProvider.EOS_TOKEN_DEFAULT,
                      cost_func=DEFAULT_COST_FUNC,
                      rnn_cell=RNN_CELLS.GRU,
                      verbose=True):

        graph = self.getGraph(batch_size=batch_size, verbose=False, enc_num_units=num_units,
                              dec_num_units=num_units, input_len=input_len,

                              rnn_cell=rnn_cell(activation_rnn),
                              target_len=target_len, cost_func=cost_func, eos_token=eos_token,
                              learning_rate=learning_rate, lamda2=lamda2, batch_norm_enabled=batch_norm_enabled,
                              decoder_first_input=decoder_first_input, rnn_hidden_dim=rnn_hidden_dim,
                              mobile_attrs_dim=mobile_attrs_dim)
        # input_keep_prob=input_keep_prob, hidden_keep_prob=hidden_keep_prob,

        train_data = PriceHistoryMobAttrsDateDataProvider(npz_path=npz_path, batch_size=batch_size, rng=self.rng,
                                                          eos_token=eos_token, with_EOS=self.with_EOS,
                                                          which_set='train')
        # during cross validation we execute our experiment multiple times and we get a score at the end
        # so this means that we need to retrain the model one final time in order to output the predictions
        # from this training procedure
        preds_dp = PriceHistoryMobAttrsDateDataProvider(npz_path=npz_path, batch_size=batch_size, rng=self.rng,
                                                        shuffle_order=False, eos_token=eos_token,
                                                        with_EOS=self.with_EOS,
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

            # let's use a simple linear progression for now
            if isinstance(dec_inputs_part, bool):
                dec_ins_percent_usages = np.ones(epochs) if dec_inputs_part else np.zeros(epochs)
            else:
                first_part = dec_inputs_part  # epochs // 4 <-- very quick pretraining
                dec_ins_percent_usages = np.concatenate((np.linspace(1., 0., first_part), np.zeros(epochs - first_part)))

            for dec_ins_percent_usage, epoch in zip(dec_ins_percent_usages, range(epochs)):
                train_error, runTime = getRunTime(
                    lambda:
                    self.trainEpoch(
                        sess=sess,
                        data_provider=train_data,
                        extraFeedDict={
                            self.is_training: True,
                            self.keep_prob_rnn_out: keep_prob_rnn_out,
                            self.keep_prob_readout: keep_prob_readout,
                            self.dec_ins_percent_usage: dec_ins_percent_usage,
                        }
                    )
                )

                if np.isnan(train_error):
                    raise Exception('do something with your learning rate because it is extremely high')

                if verbose:
                    # print 'EndEpoch%02d(%.3f secs):err(train)=%.4f,acc(train)=%.2f,err(valid)=%.2f,acc(valid)=%.2f, ' % \
                    #       (epoch + 1, runTime, train_error, train_accuracy, valid_error, valid_accuracy)
                    print 'End Epoch %02d (%.3f secs): err(train) = %.6f, current dec_ins_percent_usage: %.2f' % (
                        epoch + 1, runTime, train_error, dec_ins_percent_usage)

                dyn_stats.gatherStats(train_error=train_error)

                if verbose:
                    preds_dict, test_error = self.getPredictions(batch_size=batch_size, data_provider=preds_dp,
                                                                 sess=sess)

                    targets = preds_dp.targets
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
                 # TODO in this version we are building it full length and then we are going to improve it (trunc backprop len)
                 batch_size,
                 enc_num_units,
                 dec_num_units,
                 input_len,
                 target_len,
                 rnn_hidden_dim,
                 mobile_attrs_dim,
                 decoder_first_input=DECODER_FIRST_INPUT.ZEROS,
                 batch_norm_enabled=BATCH_NORM_ENABLED_BY_DEFAULT,
                 # TODO not cover all cases yet, need to include the other batch norm layers
                 lamda2=DEFAULT_ARR_LAMDA2,
                 eos_token=PriceHistoryMobAttrsDateDataProvider.EOS_TOKEN_DEFAULT,
                 rnn_cell=RNN_CELLS.GRU(DEFAULT_ACTIVATION_RNN),
                 cost_func=DEFAULT_COST_FUNC,
                 learning_rate=ADAM_DEFAULT_LEARNING_RATE,  # default of Adam is 1e-3
                 verbose=True):
        # momentum = 0.5
        # tf.reset_default_graph() #kind of redundant statement
        output_seq_len = target_len + self.EOS_TOKEN_LEN

        predictions = None
        loss = None
        error = None

        L2regs = []

        graph = tf.Graph()  # create new graph

        with graph.as_default():
            with tf.name_scope('parameters'):
                self.is_training = tf.placeholder(tf.bool, name="is_training")

                dec_ins_percent_usage = tf.placeholder_with_default(
                    tf.constant(self.DECODER_INPUTS_DEFAULT_PERCENTAGE_USAGE, dtype=self.dtype),
                    shape=tf.TensorShape([]), name="use_decoder_inputs")
                self.dec_ins_percent_usage = dec_ins_percent_usage

            with tf.name_scope('data'):
                inputs = tf.placeholder(dtype=self.dtype,
                                        shape=(batch_size, input_len, self.FEATURE_LEN), name="inputs")

                mobile_attrs = tf.placeholder(dtype=self.dtype,
                                              shape=(batch_size, self.MOBILE_ATTRS_LEN), name="mobile_attrs")

                targets = tf.placeholder(dtype=self.dtype, shape=(batch_size, output_seq_len), name="targets")

                decoder_inputs = tf.reshape(targets, shape=targets.get_shape().concatenate(
                    tf.TensorShape([self.TARGET_FEATURE_LEN])), name="decoder_inputs")
                self.decoder_inputs = decoder_inputs

                # so here we have the date information for the current output, obviously this should be the same date as the output
                # But how should we treat the case with the EOS (end of sequence) ? What kind of date information will be
                # provided for that case?
                # It could be something invalid like setting everything to -1 OR it could make sense to just be the next day
                # since we are not currently working with the EOS solution we are going to not bother with this part for now
                decoder_extra_inputs = tf.placeholder(dtype=self.dtype,
                                                      shape=(batch_size, output_seq_len, self.DATE_FEATURE_LEN),
                                                      name="decoder_extra_inputs")
                self.decoder_extra_inputs = decoder_extra_inputs

            with tf.name_scope('inputs'):
                # unpack matrix into 1 dim array
                inputs_series = tf.unstack(inputs, axis=1)

                if verbose:
                    print len(inputs_series)
                    print inputs_series[0]  # shape: (batch_size, 1+6)
                    print

            # now here we need to include a batch normalization layer but it should be repeated for each input
            # with tf.name_scope('batched_inputs'):
            #     if batch_norm_enabled:
            #         # if you comment then you bypass it, lets see if it works with the current modifications
            #         batch_normer = BatchNormer(bnId=0, inputs=inputs_series[0])
            #
            #         processed_inputs_series = [
            #             batch_normer.batch_norm_wrapper(inputs=cur_input, is_training=is_training)
            #             for cur_input in inputs_series]
            #     else:
            #         processed_inputs_series = inputs_series

            # NOTE that we are choosing to use no batch norm or dropout in our input layers.
            # Why? We are not using batch norm because we have already done some standaridization of the inputs
            # We are not using dropout because it can be seen as a way to do data augmentation by providing alternating
            # versions of our inputs to the model but this has not proved to be helpful for our inputs.
            # Dropout seems to not be helpful for our price history input stream because it breaks the pattern that the encoder
            # is trying to capture
            # Dropout seems to not be helpful for our mobile static attributes because these contain a lot of one hot encoded
            # attributes which will be severely (in an intuitive level) be affected by dropout functionality

            with tf.name_scope('mobile_attrs_layer'):
                mobile_attrs_processed, mobile_attrs_regularizer = fully_connected_layer_with_batch_norm_and_l2(
                    fcId=0, inputs=mobile_attrs, input_dim=self.MOBILE_ATTRS_LEN, output_dim=mobile_attrs_dim,
                    lamda2=lamda2[0], is_training=self.is_training, nonlinearity=tf.nn.elu
                )

                L2regs.append(mobile_attrs_regularizer)

            with tf.name_scope('encoder_rnn_layer'):
                # don't really care for encoder outputs, but only for its final state
                # the encoder consumes all the input to get a sense of the trend of price history
                _, encoder_final_state = rnn.static_rnn(cell=rnn_cell(num_units=enc_num_units),
                                                        inputs=inputs_series,
                                                        initial_state=None,
                                                        # TODO when using trunc backprop this should not be zero
                                                        dtype=self.dtype)

                if verbose:
                    print encoder_final_state
                    print

            with tf.name_scope('rnn_processed_dense'):
                self.keep_prob_rnn_out = tf.placeholder_with_default(  # default is useful for validation
                    tf.constant(self.DEFAULT_KEEP_PROB, dtype=self.dtype),
                    shape=tf.TensorShape([]))

                # first layer takes the output of the decoder RNN and process it into some other dimensionality, let's call rnn hidden dim
                WW = generate_weights_var(ww_id='rnn_processed', input_dim=dec_num_units, output_dim=rnn_hidden_dim,
                                          dtype=self.dtype)
                L2regs.append(lamda2[1] * tf.nn.l2_loss(WW))
                bb = tf.Variable(np.zeros(rnn_hidden_dim), dtype=self.dtype, name='bias_rnn_processed')

                rnn_processed_batch_normer = BatchNormer(bnId='rnn_processed', inputs_or_outputDim=rnn_hidden_dim)

            with tf.name_scope('readout_affine'):
                self.keep_prob_readout = tf.placeholder_with_default(  # default is useful for validation
                    tf.constant(self.DEFAULT_KEEP_PROB, dtype=self.dtype),
                    shape=tf.TensorShape([]))

                readout_WW = generate_weights_var(ww_id='readout', input_dim=rnn_hidden_dim + mobile_attrs_dim,
                                                  output_dim=self.TARGET_FEATURE_LEN, dtype=self.dtype)
                L2regs.append(lamda2[2] * tf.nn.l2_loss(readout_WW))
                readout_bb = tf.Variable(np.zeros(self.TARGET_FEATURE_LEN), dtype=self.dtype, name='bias_readout')

            with tf.variable_scope('decoder_rnn_layer'):
                eos_token_tensor = tf.constant(np.ones(shape=(batch_size, 1)) * eos_token,
                                               dtype=tf.float32, name='eos_token')
                # if we are not working with end of sequence architecture then we use as initial input to the decoder the
                # last value of the inputs
                # initial_inputs = eos_token_tensor if self.with_EOS else inputs_series[-1]
                if self.with_EOS:
                    initial_inputs = eos_token_tensor
                else:
                    # inputs_series[-1]  # batch size, 1+6
                    last_input = inputs_series[-1][:, :1]  # batch size, 1
                    # BE VERY CAREFUL WITH THIS CONVENTION: the first one should be the price info
                    # check merge_date_info in PriceHistory27DatasetGenerator to make sure

                    if decoder_first_input == self.DECODER_FIRST_INPUT.PREVIOUS_INPUT:
                        initial_inputs = last_input
                    elif decoder_first_input == self.DECODER_FIRST_INPUT.ZEROS:
                        initial_inputs = tf.zeros_like(last_input)
                    else:
                        raise Exception('decoder first input configuration is unknown: {}'.format(decoder_first_input))
                        # print "WITH PREVIOUS INPUT"

                variables = {
                    'rnn_processed_layer': {
                        'WW': WW,
                        'bb': bb,
                        'batch_norm': rnn_processed_batch_normer
                    },
                    'mobile_attrs_processed': mobile_attrs_processed,
                    'readout': {
                        'WW': readout_WW,
                        'bb': readout_bb,
                    }
                }

                decoder_output_tensor_array, decoder_final_state, decoder_final_loop_state = tf.nn.raw_rnn(
                    cell=rnn_cell(num_units=dec_num_units),
                    loop_fn=self.get_loop_fn(
                        encoder_final_state=encoder_final_state,
                        initial_inputs=initial_inputs,
                        batch_size=batch_size,
                        variables=variables,
                        static_seq_len=output_seq_len,
                        target_len=output_seq_len,
                        verbose=verbose,
                    ))
                # del decoder_output_tensor_array, decoder_final_state #not interesting
                predictions = decoder_final_loop_state
                # predictions = batchNormWrapper(bnId=2, inputs=decoder_final_loop_state, is_training=self.is_training) #do NOT

                if verbose:
                    print "decoder_final_loop_state"
                    print decoder_final_loop_state
                    print

            with tf.name_scope('error'):
                if cost_func == self.COST_FUNCS.HUBER_LOSS:
                    losses = huber_loss(y_true=targets, y_pred=predictions)  # both have shape: (batch_size, target_len)
                elif cost_func == self.COST_FUNCS.MSE:
                    losses = tf.squared_difference(predictions, targets)  # tfMSE(outputs=predictions, targets=targets)
                else:
                    raise Exception("invalid or non supported cost function")

                if verbose:
                    print losses
                    print

                loss = tf.reduce_mean(losses)
                # fix error to exclude the EOS from the reported error calculation BUT keep it inside for minimization
                if self.with_EOS:  # fix error to exclude the EOS from the error calculation
                    mask = np.ones(shape=losses.get_shape())
                    mask[:, -1:] = 0
                    masked_losses = losses * tf.constant(mask, dtype=tf.float32)

                    error = tf.reduce_mean(masked_losses)
                else:
                    error = loss

                if verbose:
                    print loss
                    print error
                    print

            with tf.name_scope('training_step'):
                for L2reg in L2regs:  # just add the L2 regularizers
                    loss += L2reg

                train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

            init = tf.global_variables_initializer()

        self.init = init
        self.inputs = inputs
        self.targets = targets
        self.mobile_attrs = mobile_attrs
        self.error = error
        self.train_step = train_step
        self.predictions = predictions
        # self.keep_prob_hidden = keep_prob_hidden

        return graph

    def pass_decoder_out_through_vars(self, layer, tfvars, verbose):
        """variables must be defined out of here due to restrictions with the while loop"""

        # num units  ->  rnn_hidden_dim
        rnn_processed_layer_vars = tfvars['rnn_processed_layer']

        layer_prob = tf.nn.dropout(layer, self.keep_prob_rnn_out, seed=self.SEED)

        rnn_processed_affine = tf.add(tf.matmul(layer_prob, rnn_processed_layer_vars['WW']),
                                      rnn_processed_layer_vars['bb'])
        rnn_processed_batch_norm = rnn_processed_layer_vars['batch_norm'].batch_norm_wrapper(rnn_processed_affine,
                                                                                             self.is_training)
        rnn_processed_layer = tf.nn.elu(rnn_processed_batch_norm, name="dense_hidden_rnn")

        mobile_attrs_processed = tfvars['mobile_attrs_processed']

        # print rnn_processed_layer.get_shape()
        # print mobile_attrs_processed.get_shape()

        # shape: (batch_size, rnn_processed_layer dim size +  mobile_attrs_processed dim size
        # the order of the concatanation here does not (should not) matter
        readout_inputs = tf.concat((rnn_processed_layer, mobile_attrs_processed), axis=1, name="readout_input")

        readout_prob = tf.nn.dropout(readout_inputs, self.keep_prob_readout, seed=self.SEED)

        if verbose:
            print "readout inputs"
            print readout_inputs
            print

        # affine_layer = tf.constant(np.zeros((100, 1)), dtype=tf.float32)
        readout_layer = tfvars['readout']
        affine_layer = tf.add(tf.matmul(readout_prob, readout_layer['WW']), readout_layer['bb'], name='readout')

        if verbose:
            print 'readout'
            print affine_layer
            print

        return affine_layer

    def get_loop_fn(self, encoder_final_state, initial_inputs, batch_size, variables, static_seq_len, target_len,
                    verbose):
        def loop_fn(time, previous_output, previous_state, previous_loop_state):
            # inputs:  time, previous_cell_output, previous_cell_state, previous_loop_state
            # outputs: elements_finished, input, cell_state, output, loop_state

            if previous_state is None:  # time == 0
                assert previous_output is None
                return self.loop_fn_initial(encoder_final_state=encoder_final_state, initial_inputs=initial_inputs,
                                            batch_size=batch_size, target_len=target_len)
            else:
                return self.loop_fn_transition(time=time, previous_cell_output=previous_output,
                                               previous_cell_state=previous_state,
                                               batch_size=batch_size, variables=variables,
                                               static_seq_len=static_seq_len, target_len=target_len,
                                               previous_loop_state=previous_loop_state, verbose=verbose)

        return loop_fn

    def loop_fn_initial(self, encoder_final_state, initial_inputs, batch_size, target_len):
        time = 0  # this is true for loop fn initial

        # https://www.tensorflow.org/api_docs/python/tf/nn/raw_rnn
        # here we have a static rnn case so all length so be taken into account,
        # so I guess they are all False always
        # From documentation: a boolean Tensor of shape [batch_size]
        initial_elements_finished = self.all_elems_non_finished(batch_size=batch_size)

        # self.decoder_extra_inputs #batch_size, output_seq_len(30), self.DATE_FEATURE_LEN(6)
        initial_input = tf.concat(values=(
            initial_inputs,  # batch_size, 1
            self.decoder_extra_inputs[:, time, :]  # becomes of shape batch_size, 6
        ), axis=1)

        initial_cell_state = encoder_final_state

        initial_cell_output = None
        # give it the shape that we want but how exactly ???:
        # initial_cell_output = tf.Variable(np.zeros(shape=(batch_size, self.TARGET_FEATURE_LEN)), dtype=tf.float32)

        # initial_loop_state = None  # we don't need to pass any additional information
        # initial_loop_state = tf.Variable(np.zeros((batch_size, None)), dtype=self.dtype)
        initial_loop_state = tf.zeros(shape=(batch_size, target_len), dtype=self.dtype)

        return (initial_elements_finished,
                initial_input,
                initial_cell_state,
                initial_cell_output,
                initial_loop_state)

    def loop_fn_transition(self, time, previous_cell_output, previous_cell_state, batch_size, variables,
                           static_seq_len, previous_loop_state, verbose, target_len):
        """note that the matrix W is going to be shared among outputs"""
        # print "previous cell output!"
        # print previous_cell_output
        # print

        if verbose:
            print "time"
            print time
            print

        # finished = self.all_elems_finished(batch_size=batch_size,
        #                                    finished=time - 1 >= static_seq_len)  # (time >= decoder_lengths)
        # finished = self.all_elems_finished(batch_size=batch_size,
        #                                    finished=time - 1 >= static_seq_len)  # (time >= decoder_lengths)
        finished = time >= self.get_seq_len_tensor(batch_size=batch_size, static_seq_len=static_seq_len)
        the_end = tf.reduce_all(finished)

        # this operation produces boolean tensor of [batch_size] defining if corresponding sequence has ended

        # this is always false in our case so just comment next two lines
        # finished = tf.reduce_all(elements_finished)  # -> boolean scalar
        # input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
        # next_input = tf.add(tf.matmul(previous_cell_output, WW), bb)

        batch_preds = self.pass_decoder_out_through_vars(previous_cell_output, tfvars=variables, verbose=verbose)
        # next_input = batch_preds

        # this works whether we have EOS or not
        # time = 1 we pass 0th dec input, on time=2 we pass 1st dec input etc.
        from_targets = self.decoder_inputs[:, time - 1]

        next_price_input = (from_targets * self.dec_ins_percent_usage) + (
            batch_preds * (1 - self.dec_ins_percent_usage))

        # becomes of shape batch_size, 6, also recall time here is 1, 2, ...
        cur_dec_extra_ins = tf.cond(the_end, lambda: tf.zeros(shape=(batch_size, self.DATE_FEATURE_LEN)),
                                    lambda: self.decoder_extra_inputs[:, time, :])

        next_input = tf.concat(values=(
            next_price_input,  # batch_size, 1
            cur_dec_extra_ins,
        ), axis=1)

        # print "next input!"
        # print next_input
        # print

        next_cell_state = previous_cell_state
        # emit_output = tf.identity(next_input),
        emit_output = previous_cell_output

        # next_loop_state = None  # we don't need to pass any additional information
        # with tf.control_dependencies([next_input]):

        # outs = tf.Variable(previous_loop_state, dtype=self.dtype) <-- cannot assign variables in while loop
        # outs[:, time].assign(next_input) <--does not work with non variables
        # print "next in"
        # print next_input
        # next_loop_state = tf.concat(values=(previous_loop_state, next_input),
        #                             axis=1)  # we don't need to pass any additional information <-- cannot have variable length
        # next_loop_state.set_shape((50, time+1))

        my_out = tf.concat(values=(previous_loop_state[:, :time - 1], batch_preds, previous_loop_state[:, time:]),
                           axis=1)

        next_loop_state = my_out  # tf.cond(the_end, lambda: previous_loop_state, lambda: my_out) #this was wrong implementation
        next_loop_state.set_shape((batch_size, target_len))

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

    def getPredictions(self, sess, data_provider, batch_size, extraFeedDict=None):
        if extraFeedDict is None:
            extraFeedDict = {}

        assert data_provider.data_len % batch_size == 0  # provider can support and intermediate values

        total_error = 0.

        instances_order = data_provider.current_order

        target_len = data_provider.targets.shape[1]

        all_predictions = np.zeros(shape=(data_provider.data_len, target_len))

        for inst_ind, (input_batch, mobile_attrs_batch, target_batch, dec_extra_ins) in enumerate(data_provider):
            cur_error, cur_preds = sess.run(
                [self.error, self.predictions],
                feed_dict=merge_dicts({self.inputs: input_batch,
                                       self.mobile_attrs: mobile_attrs_batch,
                                       self.targets: target_batch,
                                       self.decoder_extra_inputs: dec_extra_ins,
                                       self.is_training: False,
                                       }, extraFeedDict))

            assert np.all(instances_order == data_provider.current_order)

            all_predictions[inst_ind * batch_size: (inst_ind + 1) * batch_size, :] = cur_preds[:,
                                                                                     :-1] if self.with_EOS else cur_preds

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

        for step, (input_batch, mobile_attrs_batch, target_batch, dec_extra_ins) in enumerate(data_provider):
            feed_dic = merge_dicts({self.inputs: input_batch,
                                    self.mobile_attrs: mobile_attrs_batch,
                                    self.targets: target_batch,
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

        for step, (input_batch, mobile_attrs_batch, target_batch, dec_extra_ins) in enumerate(data_provider):
            feed_dic = merge_dicts({self.inputs: input_batch,
                                    self.mobile_attrs: mobile_attrs_batch,
                                    self.targets: target_batch,
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
