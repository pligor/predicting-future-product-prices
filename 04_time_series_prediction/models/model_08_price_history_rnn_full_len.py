from __future__ import division

import numpy as np
import tensorflow as tf

from cost_functions.huber_loss import huberLoss
from data_providers.price_history_full_len_data_provider import PriceHistoryFullLenDataProvider
from mylibs.jupyter_notebook_helper import DynStats, getRunTime
# from model_selection.cross_validator import CrossValidator
from interfaces.neural_net_model_interface import NeuralNetModelInterface
from tensorflow.contrib import rnn

from mylibs.tf_helper import tfMSE, trainEpoch


class PriceHistoryRnnFullLen(NeuralNetModelInterface):
    FEATURE_LEN = 1

    def __init__(self, rng, dtype, config):
        # super(ShiftFunc, self).__init__(random_state=rng, data_provider_class=ProductPairsDataProvider)
        super(PriceHistoryRnnFullLen, self).__init__()

        self.rng = rng
        self.dtype = dtype
        self.config = config
        self.train_data = None
        self.valid_data = None
        self.init = None
        self.error = None
        self.inputs = None
        self.outputs = None
        self.targets = None
        self.train_step = None

    class COST_FUNCS(object):
        HUBER_LOSS = "huberloss"
        MSE = 'mse'

    class RNN_CELLS(object):
        BASIC_RNN = tf.contrib.rnn.BasicRNNCell  # "basic_rnn"

        @staticmethod
        def LSTM(num_units):
            return tf.contrib.rnn.BasicLSTMCell(num_units=num_units, state_is_tuple=False)  # "lstm"

        GRU = tf.contrib.rnn.GRUCell  # "gru"

    def run(self, npz_path, epochs, batch_size, state_size, input_len, target_len, preds_gather_enabled=True,
            cost_func=COST_FUNCS.HUBER_LOSS, rnn_cell=RNN_CELLS.BASIC_RNN):

        graph = self.getGraph(batch_size=batch_size, verbose=False, state_size=state_size, input_len=input_len,
                              rnn_cell=rnn_cell, target_len=target_len, cost_func=cost_func)

        train_data = PriceHistoryFullLenDataProvider(npz_path=npz_path, batch_size=batch_size, rng=self.rng)

        # preds_dp = PriceHistoryStaticDataProvider(npz_path=npz_path, batch_size=batch_size, shuffle_order=False,
        #                                           truncated_backprop_len=truncated_backprop_len) if preds_gather_enabled else None

        stats = self.train_validate(train_data=train_data, valid_data=None, graph=graph, epochs=epochs,
                                    preds_gather_enabled=preds_gather_enabled)

        return stats

    def train_validate(self, train_data, valid_data, **kwargs):
        graph = kwargs['graph']
        epochs = kwargs['epochs']
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
                    trainEpoch(
                        inputs=self.inputs,
                        targets=self.targets,
                        sess=sess,
                        data_provider=train_data,
                        train_step=self.train_step,
                        error=self.error)
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

        if verbose:
            print

        return dynStats

    def getGraph(self,
                 batch_size,
                 state_size,
                 input_len,
                 target_len,
                 rnn_cell=RNN_CELLS.BASIC_RNN,
                 cost_func=COST_FUNCS.HUBER_LOSS,
                 learningRate=1e-3,  # default of Adam is 1e-3
                 lamda2=1e-2,
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
                                        shape=(batch_size, input_len, self.FEATURE_LEN), name="inputs")

                targets = tf.placeholder(dtype=self.dtype, shape=(batch_size, target_len), name="targets")

            with tf.name_scope('inputs'):
                # unpack matrix into 1 dim array
                inputs_series = tf.unstack(inputs, axis=1)

                if verbose:
                    print len(inputs_series)
                    print inputs_series[0]  # shape: (5,)
                    print

            with tf.name_scope('rnn_layer'):
                outputs, _ = rnn.static_rnn(cell=rnn_cell(num_units=state_size), inputs=inputs_series, dtype=self.dtype)

                if verbose:
                    print len(outputs)
                    print outputs[0]
                    print

                target_outputs = outputs[-target_len:]

            with tf.name_scope('readout_layer'):
                target_feature_len = 1  # this is considered static, because we are always trying to predict the price of a single product
                WW = tf.Variable(self.rng.randn(state_size, target_feature_len), dtype=self.dtype, name='weights')
                bb = tf.Variable(np.zeros(target_feature_len), dtype=self.dtype, name='bias')
                readouts = [tf.matmul(output, WW) + bb for output in target_outputs]

                if verbose:
                    print len(readouts)
                    print readouts[0]
                    print

            with tf.name_scope('predictions'):
                predictions = tf.reshape(tf.stack(readouts, axis=1), shape=(batch_size, target_len))
                if verbose:
                    # print predictions_series[0]
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
        self.outputs = predictions

        # self.keep_prob_input = keep_prob_input
        # self.keep_prob_hidden = keep_prob_hidden

        return graph
