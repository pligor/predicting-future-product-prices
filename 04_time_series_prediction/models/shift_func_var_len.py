from __future__ import division

import numpy as np
import tensorflow as tf

from data_providers.binary_shifter_varlen_data_provider import BinaryShifterVarLenDataProvider
from mylibs.jupyter_notebook_helper import DynStats, getRunTime
# from model_selection.cross_validator import CrossValidator
from interfaces.neural_net_model_interface import NeuralNetModelInterface
from tensorflow.contrib import rnn
from tensorflow.contrib import learn
from rnn_model import DynRnnModel


class ShiftFuncVarLen(NeuralNetModelInterface, DynRnnModel):
    """https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html"""

    FEATURE_LEN = 1
    num_classes = 2

    def __init__(self, rng, dtype, config, batch_size):
        # super(ShiftFunc, self).__init__(random_state=rng, data_provider_class=ProductPairsDataProvider)
        super(ShiftFuncVarLen, self).__init__()

        self.rng = rng
        self.dtype = dtype
        self.config = config
        self.batch_size = batch_size
        self._initDynRnnModel()

        self.train_data = None
        self.valid_data = None
        self.init = None
        self.error = None
        self.accuracy = None
        self.inputs = None
        self.targets = None
        self.train_step = None
        self.last_state = None
        self.sequence_lens = None
        self.sequence_len_mask = None
        self.outputs = None

    def run(self, num_instances, epochs, series_max_len, echo_step, state_size):
        graph = self.getGraph(batch_size=self.batch_size, state_size=state_size,
                              num_classes=self.num_classes, verbose=False, series_max_len=series_max_len)

        stats = self.train_validate(train_data=BinaryShifterVarLenDataProvider(
            N_instances=num_instances, series_max_len=series_max_len, echo_step=echo_step,
        ), valid_data=None, graph=graph, epochs=epochs)

        return stats

    def train_validate(self, train_data, valid_data, **kwargs):
        graph = kwargs['graph']
        epochs = kwargs['epochs']  # if 'epochs' in kwargs.keys() else 10
        verbose = kwargs['verbose'] if 'verbose' in kwargs.keys() else True

        if verbose:
            print "epochs: %d" % epochs

        with tf.Session(graph=graph, config=self.config) as sess:
            sess.run(self.init)  # sess.run(tf.initialize_all_variables())

            dynStats = DynStats()

            for epoch in range(epochs):
                (train_error, train_accuracy), runTime = getRunTime(
                    lambda:
                    self.trainEpoch(
                        inputs=self.inputs,
                        targets=self.targets,
                        sess=sess,
                        train_data=train_data,
                        train_step=self.train_step,
                        error=self.error,
                        accuracy=self.accuracy,
                        sequence_lens=self.sequence_lens,
                        sequence_len_mask=self.sequence_len_mask,
                        extraFeedDict={
                            # self.sequence_lens: np.repeat(self.truncated backprop len, self.batch_size),
                            # self.keep_prob_input: input_keep_prob,
                            # self.keep_prob_hidden: hidden_keep_prob,
                            # self.is_training: True
                        })
                )

                # if (epoch + 1) % 1 == 0:
                # valid_error, valid_accuracy = validateEpoch(
                #     inputs=self.inputs,
                #     targets=self.targets,
                #     sess=sess,
                #     valid_data=valid_data,
                #     error=self.error,
                #     accuracy=self.accuracy,
                #     extraFeedDict={self.is_training: False},
                #     # keep_prob_keys=[self.keep_prob_input, self.keep_prob_hidden]
                # )
                valid_error, valid_accuracy = 0, 0

                if verbose:
                    print 'EndEpoch%02d(%.3f secs):err(train)=%.4f,acc(train)=%.2f,err(valid)=%.2f,acc(valid)=%.2f, ' % \
                          (epoch + 1, runTime, train_error, train_accuracy, valid_error, valid_accuracy)

                dynStats.gatherStats(train_error, train_accuracy, valid_error, valid_accuracy)

        if verbose:
            print

        return dynStats.stats  # , dynStats.keys

    def getGraph(self,
                 batch_size,
                 num_classes,
                 series_max_len,
                 state_size,
                 learningRate=0.3,  # 1e-3,  # default of Adam is 1e-3
                 lamda2=1e-2,
                 verbose=True):
        # momentum = 0.5
        # tf.reset_default_graph() #kind of redundant statement
        if verbose:
            print "lamda2: %f" % lamda2
            print "learning rate: %f" % learningRate

        graph = tf.Graph()  # create new graph

        with graph.as_default():
            with tf.name_scope('rnn_cell'):
                rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=state_size)

            with tf.name_scope('data'):
                inputs = tf.placeholder(dtype=self.dtype,
                                        shape=(batch_size, series_max_len, self.FEATURE_LEN),
                                        name="batch_x_placeholder")

                targets = tf.placeholder(dtype=tf.int32,
                                         shape=(batch_size, series_max_len),
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

                # [vocab_size, state_size]
                # [batch_size, max_time]

                # this is what we had
                # [5, 15, 1]
                # and now we want to have
                # [5, 600, 1] #so we MUST KNOW the maximum length, here is 600, we are just padding with zeros
                rnn_outputs, last_state = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                            inputs=inputs,
                                                            initial_state=init_state,
                                                            dtype=self.dtype,
                                                            sequence_length=sequence_lens)
                if verbose:
                    print "rnn_outputs:"
                    print rnn_outputs
                    print

                rnn_outputs_flat = tf.reshape(rnn_outputs, (-1, state_size))

            with tf.name_scope('readout_layer'):
                WW = tf.Variable(self.rng.randn(state_size, num_classes), dtype=self.dtype, name='WW')
                bb = tf.Variable(np.zeros(num_classes), dtype=self.dtype, name='bb')

                # calculate and minimize
                # logits means logistic transform
                # these logits (num_classes =2) so it will predict if it is zero or one
                # logits_series = [tf.matmul(output, WW) + bb for output in rnn_outputs]
                logits = tf.matmul(rnn_outputs_flat, WW) + bb

                if verbose:
                    print logits
                    print

            with tf.name_scope('predictions'):
                predictions = tf.nn.softmax(logits=logits)

            with tf.name_scope('labels'):
                labels = tf.reshape(targets, shape=(-1,))

                if verbose:
                    print labels
                    print

            with tf.name_scope('error_or_loss'):
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                losses_reshaped = tf.reshape(losses, shape=(batch_size, series_max_len))
                # print losses_reshaped

                # losses that doesn't matter, out of the range of the sequence length of each sequence, should be set to zero
                # losses_fixed = losses * tf.cast(tf.reshape(sequence_len_mask, shape=(-1,)), self.dtype)
                losses_fixed = losses_reshaped * tf.cast(sequence_len_mask, self.dtype)
                if verbose:
                    print "losses fixed"
                    print losses_fixed
                    print

                # but now we should NOT just take the mean because each sequence of the batch has different length, so
                # we need to divide by the appropriate length

                loss_per_sequence = tf.reduce_sum(losses_fixed, axis=1)
                if verbose:
                    print "loss per seq"
                    print loss_per_sequence
                    print

                loss_per_seq_normalized = loss_per_sequence / tf.cast(sequence_lens, self.dtype)
                if verbose:
                    print "loss normalized"
                    print loss_per_seq_normalized
                    print

                error = tf.reduce_mean(loss_per_seq_normalized)

                if verbose:
                    print losses
                    print error

            with tf.name_scope('accuracy'):
                accuracy = error

            with tf.name_scope('training_step'):
                train_step = tf.train.AdagradOptimizer(learning_rate=learningRate).minimize(error)

            init = tf.global_variables_initializer()

        self.init = init
        self.inputs = inputs
        self.targets = targets
        self.last_state = last_state
        self.sequence_lens = sequence_lens
        self.error = error
        self.accuracy = accuracy
        self.train_step = train_step
        self.outputs = predictions
        self.sequence_len_mask = sequence_len_mask

        return graph
