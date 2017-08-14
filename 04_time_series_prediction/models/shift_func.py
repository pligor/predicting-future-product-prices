from __future__ import division

import numpy as np
import tensorflow as tf

from nn_io.binary_shifter_data_provider import BinaryShifterDataProvider
from mylibs.jupyter_notebook_helper import DynStats, getRunTime
# from model_selection.cross_validator import CrossValidator
from interfaces.neural_net_model_interface import NeuralNetModelInterface
from tensorflow.contrib import rnn
from tensorflow.contrib import learn
from rnn_model import RnnModel


class ShiftFunc(NeuralNetModelInterface, RnnModel):
    INPUT_DIM = 278
    FEATURE_LEN = 1
    num_classes = 2

    def __init__(self, rng, dtype, config, truncated_backprop_len, state_size, batch_size):
        # super(ShiftFunc, self).__init__(random_state=rng, data_provider_class=ProductPairsDataProvider)
        super(ShiftFunc, self).__init__()

        self.outputs = None
        self.rng = rng
        self.dtype = dtype
        self.config = config

        self.train_data = None
        self.valid_data = None
        self.init = None
        self.error = None
        self.accuracy = None
        self.inputs = None
        self.targets = None
        self.train_step = None
        self.init_state = None
        self.last_state = None

        self.truncated_backprop_len = truncated_backprop_len
        self.state_size = state_size
        self.batch_size = batch_size

    def run(self, num_instances, total_series_length, epochs, truncated_backprop_len, echo_step):
        graph = self.getGraph(batch_size=self.batch_size,
                              num_classes=self.num_classes, verbose=False)

        stats = self.train_validate(train_data=BinaryShifterDataProvider(
            N_instances=num_instances, total_series_length=total_series_length, echo_step=echo_step,
            truncated_backprop_len=truncated_backprop_len,
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
                        batch_size=self.batch_size,
                        state_size=self.state_size,

                        inputs=self.inputs,
                        targets=self.targets,
                        sess=sess,
                        data_provider=train_data,
                        train_step=self.train_step,
                        error=self.error,
                        accuracy=self.accuracy,
                        init_state=self.init_state,
                        last_state=self.last_state,
                        extraFeedDict={
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
                 learningRate=0.3,  # 1e-3,  # default of Adam is 1e-3
                 lamda2=1e-2,
                 inputDim=INPUT_DIM,
                 verbose=True):
        # momentum = 0.5
        # tf.reset_default_graph() #kind of redundant statement
        if verbose:
            print "lamda2: %f" % lamda2
            print "learning rate: %f" % learningRate

        graph = tf.Graph()  # create new graph

        with graph.as_default():
            with tf.name_scope('rnn_cell'):
                rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=self.state_size)

            with tf.name_scope('data'):
                batchX_placeholder = tf.placeholder(dtype=self.dtype,
                                                    shape=(batch_size, self.truncated_backprop_len, self.FEATURE_LEN),
                                                    name="batch_x_placeholder")

                batchY_placeholder = tf.placeholder(dtype=tf.int32,
                                                    shape=(batch_size, self.truncated_backprop_len, self.FEATURE_LEN),
                                                    name="batch_y_placeholder")

                # the initial state of the recurrent neural network
                init_state = tf.placeholder(dtype=self.dtype, shape=(batch_size, self.state_size),
                                            name="init_state")

            with tf.name_scope('inputs'):
                # unpack matrix into 1 dim array
                inputs_series = tf.unstack(batchX_placeholder, axis=1)

                # inputs_series = tf.split(batchX_placeholder, truncated_backprop_len, axis=1)

                # second line is better because we have to pass (batch_size, feature_len) to the RNN
                # while if you have only one feature you would get (batch_size,) <-- one less dim
                if verbose:
                    print len(inputs_series)
                    print inputs_series[0]  # shape: (5,)
                    print

            with tf.name_scope('rnn_layer'):
                outputs, last_state = rnn.static_rnn(cell=rnn_cell,
                                                     initial_state=init_state,
                                                     inputs=inputs_series,
                                                     dtype=self.dtype)
                if verbose:
                    print len(outputs)
                    print outputs[0]
                    print

            with tf.name_scope('readout_layer'):
                WW = tf.Variable(self.rng.randn(self.state_size, num_classes), dtype=self.dtype, name='W2')
                bb = tf.Variable(np.zeros(num_classes), dtype=self.dtype, name='b2')
                # calculate and minimize
                # logits means logistic transform
                # these logits (num_classes =2) so it will predict if it is zero or one
                logits_series = [tf.matmul(output, WW) + bb for output in outputs]

                if verbose:
                    print logits_series[0]
                    print

            with tf.name_scope('predictions'):
                predictions_series = [tf.nn.softmax(logits=logits) for logits in logits_series]

            with tf.name_scope('labels'):
                labels_series = tf.unstack(batchY_placeholder, axis=1)

                if verbose:
                    print len(labels_series)
                    print labels_series[0]
                    print

            with tf.name_scope('error_or_loss'):
                losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,  # labels=labels)
                                                                         labels=tf.reshape(labels, (batch_size,)))

                          for logits, labels in zip(logits_series, labels_series)]
                error = tf.reduce_mean(losses)

                if verbose:
                    print error

            with tf.name_scope('accuracy'):
                # accuracies =
                accuracy = error
                # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(targets, axis=1)),
                #                                   dtype=self.dtype))

            with tf.name_scope('training_step'):
                train_step = tf.train.AdagradOptimizer(learning_rate=learningRate).minimize(error)

            init = tf.global_variables_initializer()

        self.init = init
        self.inputs = batchX_placeholder
        self.targets = batchY_placeholder
        self.init_state = init_state
        self.last_state = last_state

        self.error = error
        self.accuracy = accuracy
        # self.is_training = is_training
        self.train_step = train_step
        self.outputs = predictions_series

        # self.keep_prob_input = keep_prob_input
        # self.keep_prob_hidden = keep_prob_hidden

        return graph
