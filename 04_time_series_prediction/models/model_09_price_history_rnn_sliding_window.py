from __future__ import division

import numpy as np
import tensorflow as tf

from data_providers.price_history_sliding_window_data_provider import PriceHistorySlidingWindowDataProvider
from mylibs.jupyter_notebook_helper import DynStats, getRunTime
from interfaces.neural_net_model_interface import NeuralNetModelInterface
from tensorflow.contrib import rnn
from cost_functions.huber_loss import huber_loss
from cost_functions.mse import tf_mse
from mylibs.py_helper import merge_dicts
from collections import OrderedDict


class PriceHistoryRnnSlidingWindow(NeuralNetModelInterface):
    FEATURE_LEN = 1
    TARGET_FEATURE_LEN = 1

    def __init__(self, rng, dtype, config):
        # super(ShiftFunc, self).__init__(random_state=rng, data_provider_class=ProductPairsDataProvider)
        super(PriceHistoryRnnSlidingWindow, self).__init__()

        self.losses_mask = None
        self.outputs = None
        self.rng = rng
        self.dtype = dtype
        self.config = config
        self.train_data = None
        self.valid_data = None
        self.init = None
        self.error = None
        self.inputs = None
        self.targets = None
        self.train_step = None
        self.init_state = None
        self.last_state = None
        self.count_inc_losses = None
        self.target_error = None

    class COST_FUNCS(object):
        HUBER_LOSS = "huberloss"
        MSE = 'mse'

    class RNN_CELLS(object):
        BASIC_RNN = tf.contrib.rnn.BasicRNNCell  # "basic_rnn"

        @staticmethod
        def LSTM(num_units):
            return tf.contrib.rnn.BasicLSTMCell(num_units=num_units, state_is_tuple=False)  # "lstm"

        GRU = tf.contrib.rnn.GRUCell  # "gru"

    def run(self, npz_path, epochs, trunc_backprop_len, batch_size, pred_len, state_size,
            preds_gather_enabled=True, cost_func=COST_FUNCS.HUBER_LOSS, rnn_cell=RNN_CELLS.BASIC_RNN):

        graph = self.getGraph(batch_size=batch_size, verbose=False, state_size=state_size,
                              trunc_backprop_len=trunc_backprop_len, rnn_cell=rnn_cell,
                              pred_len=pred_len, cost_func=cost_func)

        train_data = PriceHistorySlidingWindowDataProvider(npz_path=npz_path, batch_size=batch_size, rng=self.rng,
                                                           trunc_backprop_len=trunc_backprop_len, pred_len=pred_len)

        preds_dp = PriceHistorySlidingWindowDataProvider(npz_path=npz_path, batch_size=batch_size, shuffle_order=False,
                                                         trunc_backprop_len=trunc_backprop_len,
                                                         pred_len=pred_len) if preds_gather_enabled else None

        stats = self.train_validate(train_data=train_data, valid_data=None, graph=graph, epochs=epochs,
                                    preds_gather_enabled=preds_gather_enabled, state_size=state_size,
                                    batch_size=batch_size, preds_dp=preds_dp, pred_len=pred_len)

        return stats

    def train_validate(self, train_data, valid_data, **kwargs):
        graph = kwargs['graph']
        state_size = kwargs['state_size']
        epochs = kwargs['epochs']
        batch_size = kwargs['batch_size']
        pred_len = kwargs['pred_len']
        verbose = kwargs['verbose'] if 'verbose' in kwargs.keys() else True

        preds_dp = kwargs['preds_dp'] if 'preds_dp' in kwargs.keys() else None
        preds_gather_enabled = kwargs['preds_gather_enabled'] if 'preds_gather_enabled' in kwargs.keys() else False

        if verbose:
            print "epochs: %d" % epochs

        # min_error = float("inf")

        with tf.Session(graph=graph, config=self.config) as sess:
            sess.run(self.init)  # sess.run(tf.initialize_all_variables())

            dynStats = DynStats()

            for epoch in range(epochs):
                train_error, runTime = getRunTime(
                    lambda:
                    self.trainEpoch(
                        batch_size=batch_size,
                        state_size=state_size,
                        sess=sess,
                        data_provider=train_data)
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

            predictions_dict = self.getPredictions(batch_size=batch_size, data_provider=preds_dp,
                                                   sess=sess, state_size=state_size, pred_len=pred_len)[
                0] if preds_gather_enabled else None

        if verbose:
            print

        if preds_gather_enabled:
            return dynStats, predictions_dict
        else:
            return dynStats

    def getPredictions(self, sess, data_provider, batch_size, state_size, pred_len, extraFeedDict=None):
        if extraFeedDict is None:
            extraFeedDict = {}

        assert data_provider.data_len % batch_size == 0

        total_error = 0.

        instances_order = data_provider.current_order

        target_len = data_provider.targets.shape[1]
        all_predictions = np.zeros(shape=(data_provider.data_len, target_len, self.TARGET_FEATURE_LEN))

        cur_state = self.zeroState(batch_size=batch_size, state_size=state_size)

        def reset_batch_preds_stack():
            return np.empty(shape=(batch_size, 0, self.TARGET_FEATURE_LEN))

        cur_batch_preds_stack = reset_batch_preds_stack()

        inst_ind = 0

        for input_batch, target_batch, new_instance, count_targets_of_interest, losses_mask in data_provider:
            if new_instance:
                cur_state = self.zeroState(batch_size=batch_size, state_size=state_size)
                # take all the predictions for the current instance accumulated insert them
                # print all_predictions[step * batch_size: (step + 1) * batch_size, :, :].shape
                # print cur_batch_preds_stack.shape
                all_predictions[inst_ind * batch_size: (inst_ind + 1) * batch_size, :, :] = cur_batch_preds_stack
                cur_batch_preds_stack = reset_batch_preds_stack()
                inst_ind += 1

            target_err, _, cur_state, cur_preds = sess.run(
                [self.target_error, self.error, self.last_state, self.outputs],
                feed_dict=merge_dicts({self.inputs: input_batch,
                                       self.targets: target_batch,
                                       self.init_state: cur_state,
                                       self.count_inc_losses: count_targets_of_interest,
                                       self.losses_mask: losses_mask
                                       }, extraFeedDict))

            cur_preds_of_interest = cur_preds[:, pred_len - count_targets_of_interest:, :]
            cur_batch_preds_stack = np.concatenate((cur_batch_preds_stack, cur_preds_of_interest), axis=1)

            assert np.all(instances_order == data_provider.current_order)

            total_error += target_err

            data_provider.stackPreds(new_preds=cur_preds)

        all_predictions[inst_ind * batch_size: (inst_ind + 1) * batch_size, :, :] = cur_batch_preds_stack

        total_error /= data_provider.num_batches

        assert np.all(all_predictions != 0)  # all predictions are expected to be something else than absolute zero

        preds_dict = OrderedDict(zip(instances_order, all_predictions))

        return preds_dict, total_error

    @staticmethod
    def zeroState(batch_size, state_size):
        return np.zeros(shape=(batch_size, state_size))

    def trainEpoch(self,
                   batch_size,
                   state_size,
                   sess,
                   data_provider,
                   extraFeedDict=None):
        if extraFeedDict is None:
            extraFeedDict = {}

        train_error = 0.

        num_batches = data_provider.num_batches

        cur_state = self.zeroState(batch_size=batch_size, state_size=state_size)

        for input_batch, target_batch, new_instance, count_targets_of_interest, losses_mask in data_provider:
            # print "losses: {}".format(count_targets_of_interest)

            if new_instance:
                cur_state = self.zeroState(batch_size=batch_size, state_size=state_size)

            feed_dict = merge_dicts({self.inputs: input_batch,
                                     self.targets: target_batch,
                                     self.init_state: cur_state,
                                     self.count_inc_losses: count_targets_of_interest,
                                     self.losses_mask: losses_mask
                                     }, extraFeedDict)

            # if accuracy is None:
            _, target_err, _, cur_state, preds = sess.run(
                [self.train_step, self.target_error, self.error, self.last_state, self.outputs],
                feed_dict=feed_dict)
            # else:
            #     cur_state, _, batch_error, batch_acc = sess.run([self.last_state, self.train_step, self.error, accuracy], feed_dict=feed_dict)

            # WE WANT TO TAKE INTO ACCOUNT ONLY THE PREDICTIONS THAT ARE RELATED TO TARGET
            train_error += target_err
            # if accuracy is not None:
            #     train_accuracy += batch_acc

            # if accuracy is not None:
            #     train_accuracy += batch_acc

            data_provider.stackPreds(new_preds=preds)

        train_error /= num_batches
        # train_accuracy /= num_batches

        # if accuracy is None:
        return train_error
        # else:
        #    return train_error, train_accuracy

    def getGraph(self,
                 batch_size,
                 state_size,
                 trunc_backprop_len,
                 pred_len,  # target_len,
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

        graph = tf.Graph()  # create new graph

        with graph.as_default():
            with tf.name_scope('data'):
                inputs = tf.placeholder(dtype=self.dtype,
                                        shape=(batch_size, trunc_backprop_len, self.FEATURE_LEN), name="inputs")

                targets = tf.placeholder(dtype=self.dtype, shape=(batch_size, pred_len, self.TARGET_FEATURE_LEN),
                                         name="targets")

            # the initial state of the recurrent neural network
            init_state = tf.placeholder(dtype=self.dtype, shape=(batch_size, state_size), name="init_state")

            count_inc_losses = tf.placeholder(dtype=tf.int32, name="count_inc_losses") #kind of redundant

            losses_mask = tf.placeholder(dtype=self.dtype, shape=(batch_size, pred_len, self.TARGET_FEATURE_LEN),
                                         name="losses_mask")

            with tf.name_scope('inputs'):
                # unpack matrix into 1 dim array
                inputs_series = tf.unstack(inputs, axis=1)

                if verbose:
                    print len(inputs_series)
                    print inputs_series[0]  # shape: (5,)
                    print

            with tf.name_scope('rnn_layer'):
                rnn_outputs, last_state = rnn.static_rnn(cell=rnn_cell(num_units=state_size),
                                                         initial_state=init_state,
                                                         inputs=inputs_series,
                                                         dtype=self.dtype)
                if verbose:
                    print len(rnn_outputs)
                    print rnn_outputs[0]
                    print

                pred_outputs = rnn_outputs[-pred_len:]

            with tf.name_scope('readout_layer'):
                WW = tf.Variable(self.rng.randn(state_size, self.TARGET_FEATURE_LEN), dtype=self.dtype, name='weights')
                bb = tf.Variable(np.zeros(self.TARGET_FEATURE_LEN), dtype=self.dtype, name='bias')
                readouts = [tf.matmul(output, WW) + bb for output in pred_outputs]

                if verbose:
                    print readouts[0]
                    print

            with tf.name_scope('predictions'):
                predictions = tf.stack(readouts, axis=1)
                if verbose:
                    print predictions
                    print

            with tf.name_scope('error'):
                if cost_func == self.COST_FUNCS.HUBER_LOSS:
                    losses = huber_loss(y_true=targets, y_pred=predictions)  # both have shape: (batch_size, target_len)
                elif cost_func == self.COST_FUNCS.MSE:
                    losses = tf_mse(outputs=predictions, targets=targets)
                else:
                    raise Exception("invalid or non supported cost function")

                if verbose:
                    print losses
                    print

                # size_shape = list(losses.get_shape())
                # begin_shape = list(np.zeros(len(size_shape)).astype(np.int))
                # size_shape = (size_shape[0].value, count_inc_losses, size_shape[2].value)
                # begin_shape = (begin_shape[0], pred_len - count_inc_losses, begin_shape[2])
                # sliced_losses = tf.slice(losses, begin_shape, size_shape)
                masked_losses = losses_mask * losses
                # target_error = tf.reduce_mean(sliced_losses)
                target_error = tf.reduce_mean(masked_losses)

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
        self.init_state = init_state
        self.last_state = last_state
        self.error = error
        self.target_error = target_error
        self.train_step = train_step
        self.outputs = predictions
        self.count_inc_losses = count_inc_losses
        self.losses_mask = losses_mask

        # self.keep_prob_input = keep_prob_input
        # self.keep_prob_hidden = keep_prob_hidden

        return graph
