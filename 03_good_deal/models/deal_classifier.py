from __future__ import division

import numpy as np
import tensorflow as tf

from libs.data_providers import ProductPairsDataProvider
from mylibs.batch_norm import fully_connected_layer_with_batch_norm_and_l2
from mylibs.jupyter_notebook_helper import DynStats, getRunTime
from model_selection.cross_validator import CrossValidator
from mylibs.tf_helper import trainEpoch, validateEpoch


class DealClassifier(CrossValidator):
    """
    Notes for NOT choosing the siamese neural network
    The labels do NOT represent whether product one and or product two are of different categories but whether one is a better deal than the other
    We could say with the siamese network that comparison of two products that one is better than the other, is similar to comparison of other pairs
    where the one product is better than the other.
    And we could also say that pairs of products where one is worse than the other are similar to
    comparison of products where one is again worse than the other.
    BUT this is an optimization, it is not like we are gaining anything
    """

    INPUT_DIM = 278
    NUM_CLASSES = 2
    BATCH_SIZE = 58

    def __init__(self, rng, dtype, config):
        super(DealClassifier, self).__init__(random_state=rng, data_provider_class=ProductPairsDataProvider)
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
        self.is_training = None
        self.keep_prob_input = None
        self.keep_prob_hidden = None
        self.train_step = None

    def run(self, hidden_dim, lamda2, learning_rate, epochs, n_splits, input_keep_prob=1., hidden_keep_prob=1.):
        graph = self.getGraph(hidden_dim=hidden_dim, lamda2=lamda2, learningRate=learning_rate)

        stats_list = self.cross_validate(n_splits=n_splits, batch_size=self.BATCH_SIZE, data_provider_params={
            "rng": self.rng
        }, graph=graph, input_keep_prob=input_keep_prob, hidden_keep_prob=hidden_keep_prob, epochs=epochs)

        return stats_list

    def train_validate(self, train_data, valid_data, **kwargs):

        graph = kwargs['graph']
        input_keep_prob = kwargs['input_keep_prob'] if 'input_keep_prob' in kwargs.keys() else 1.
        hidden_keep_prob = kwargs['hidden_keep_prob'] if 'hidden_keep_prob' in kwargs.keys() else 1.
        epochs = kwargs['epochs'] if 'epochs' in kwargs.keys() else 10
        verbose = kwargs['verbose'] if 'verbose' in kwargs.keys() else True

        if verbose:
            print "epochs: %d" % epochs
            print "input_keep_prob: %f" % input_keep_prob
            print "hidden_keep_prob: %f" % hidden_keep_prob

        with tf.Session(graph=graph, config=self.config) as sess:
            sess.run(self.init)

            dynStats = DynStats()

            for epoch in range(epochs):
                (train_error, train_accuracy), runTime = getRunTime(
                    lambda:
                    trainEpoch(
                        inputs=self.inputs,
                        targets=self.targets,
                        sess=sess,
                        train_data=train_data,
                        train_step=self.train_step,
                        error=self.error,
                        accuracy=self.accuracy,
                        extraFeedDict={
                            # self.keep_prob_input: input_keep_prob,
                            # self.keep_prob_hidden: hidden_keep_prob,
                            self.is_training: True
                        })
                )

                # if (epoch + 1) % 1 == 0:
                valid_error, valid_accuracy = validateEpoch(
                    inputs=self.inputs,
                    targets=self.targets,
                    sess=sess,
                    valid_data=valid_data,
                    error=self.error,
                    accuracy=self.accuracy,
                    extraFeedDict={self.is_training: False},
                    # keep_prob_keys=[self.keep_prob_input, self.keep_prob_hidden]
                )

                if verbose:
                    print 'End epoch %02d (%.3f secs):err(train)=%.2f,acc(train)=%.2f,err(valid)=%.2f,acc(valid)=%.2f, ' % \
                          (epoch + 1, runTime, train_error, train_accuracy, valid_error, valid_accuracy)

                dynStats.gatherStats(train_error, train_accuracy, valid_error, valid_accuracy)

        if verbose:
            print

        return dynStats.stats  # , dynStats.keys

    # def validate(self, data_provider, graph, epochs=35, verbose=True):
    #     if verbose:
    #         print "epochs: %d" % epochs
    #
    #     with tf.Session(graph=graph, config=self.config) as sess:
    #         sess.run(self.init)
    #
    #         stats, keys = initStats(epochs)
    #         runTimes = []
    #
    #         for e in range(epochs):
    #             (valid_error, valid_accuracy), runTime = getRunTime(lambda: validateEpoch(
    #                 inputs=self.inputs,
    #                 targets=self.targets,
    #                 sess=sess,
    #                 valid_data=data_provider,
    #                 error=self.error,
    #                 accuracy=self.accuracy,
    #                 extraFeedDict={self.training: False},
    #                 keep_prob_keys=[self.keep_prob_input, self.keep_prob_hidden]
    #             ))
    #
    #             runTimes.append(runTime)
    #
    #             if verbose:
    #                 print 'End epoch %02d (%.3f secs): err(valid)=%.2f, acc(valid)=%.2f, ' % \
    #                       (e + 1, runTime, valid_error, valid_accuracy)
    #
    #             stats = gatherStats(e, 0., 0., valid_error, valid_accuracy, stats)
    #     if verbose:
    #         print
    #
    #     return stats, keys, runTimes

    def getGraph(self,
                 learningRate=1e-3,  # default of Adam is 1e-3
                 lamda2=1e-2,
                 inputDim=INPUT_DIM,
                 numClasses=NUM_CLASSES,
                 hidden_dim=100,
                 batch_size=BATCH_SIZE,
                 verbose=True):
        # momentum = 0.5
        # tf.reset_default_graph() #kind of redundant statement
        if verbose:
            print "lamda2: %f" % lamda2
            print "hidden dim: %d" % hidden_dim
            print "learning rate: %f" % learningRate

        graph = tf.Graph()  # create new graph

        with graph.as_default():
            with tf.name_scope('data'):
                # inputs = tf.placeholder(self.dtype, [None, inputDim], 'inputs')
                inputs = tf.placeholder(self.dtype, [batch_size, inputDim], 'inputs')
                targets = tf.placeholder(self.dtype, [batch_size, numClasses], 'targets')
                # targets = tf.placeholder(self.dtype, batch_size, 'targets')

                inputs_left, inputs_right = self.cutTensorInHalf(inputs)

                # print inputs_left.get_shape()  #58, 139
                # print inputs_right.get_shape()  #58, 139

            is_training = tf.placeholder(tf.bool, name="is_training")

            feature_len = int(inputs_left.get_shape()[1])
            assert feature_len == int(inputs_right.get_shape()[1])

            regs = []
            with tf.name_scope('data_process'):
                with tf.name_scope('worse_product'):
                    with tf.name_scope('fc_worse_00'):
                        hidden_layer, regularizer = fully_connected_layer_with_batch_norm_and_l2(fcId="worse_00",
                                                                                                 inputs=inputs_left,
                                                                                                 input_dim=feature_len,
                                                                                                 output_dim=hidden_dim,
                                                                                                 training=is_training,
                                                                                                 lamda2=lamda2)
                        regs.append(regularizer)

                    with tf.name_scope('fc_worse_01'):
                        hl_worse, regularizer = fully_connected_layer_with_batch_norm_and_l2(fcId="worse_01",
                                                                                             inputs=hidden_layer,
                                                                                             input_dim=hidden_dim,
                                                                                             output_dim=hidden_dim,
                                                                                             training=is_training,
                                                                                             lamda2=lamda2)
                        regs.append(regularizer)

                with tf.name_scope('better_product'):
                    with tf.name_scope('fc_better_00'):
                        hidden_layer, regularizer = fully_connected_layer_with_batch_norm_and_l2(fcId="better_00",
                                                                                                 inputs=inputs_right,
                                                                                                 input_dim=feature_len,
                                                                                                 output_dim=hidden_dim,
                                                                                                 training=is_training,
                                                                                                 lamda2=lamda2)
                        regs.append(regularizer)

                    with tf.name_scope('fc_better_01'):
                        hl_better, regularizer = fully_connected_layer_with_batch_norm_and_l2(fcId="better_01",
                                                                                              inputs=hidden_layer,
                                                                                              input_dim=hidden_dim,
                                                                                              output_dim=hidden_dim,
                                                                                              training=is_training,
                                                                                              lamda2=lamda2)
                        regs.append(regularizer)

            xx = self.joinTensors(left_tensor=hl_worse, right_tensor=hl_better, name='xx_up')

            joined_dim = int(xx.get_shape()[1])

            with tf.name_scope('fc_joined_00'):
                hl_joined, regularizer = fully_connected_layer_with_batch_norm_and_l2(fcId="joined_00",
                                                                                      inputs=xx,
                                                                                      input_dim=joined_dim,
                                                                                      output_dim=hidden_dim,
                                                                                      training=is_training,
                                                                                      lamda2=lamda2)
                regs.append(regularizer)

            with tf.name_scope('readout_layer'):
                logits, regularizer = fully_connected_layer_with_batch_norm_and_l2(fcId="joined_01",
                                                                                   inputs=hl_joined,
                                                                                   input_dim=hidden_dim,
                                                                                   output_dim=numClasses,
                                                                                   training=is_training,
                                                                                   lamda2=lamda2,
                                                                                   nonlinearity=tf.identity,
                                                                                   )
                regs.append(regularizer)

            with tf.name_scope('error'):
                error = tf.reduce_mean(
                    # here we could use sparse_softmax_cross_entropy_with_logits
                    # because what logits are, are like this: [0.2, 0.6, 0.1, 0.1] so we compare these logits
                    # with the targets
                    # but if the targets are always like: [0, 0, 1, 0] or [0, 1, 0, 0]
                    # (note the first would classify wrong the second correct with the above logits)
                    # then this means you always have one class exactly and better treat it as sparse (only one class is correct)
                    # This also means something else which is better. That targets don't need to be absolute, your targets could also be
                    # like this: [0.1, 0.7, 0.05, 0.15] which means that you are not exactly sure when you were collecting them

                    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)
                )

                # just add all the L2 factors in the error
                for reg in regs:
                    error += reg

            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(targets, axis=1)),
                                                  dtype=self.dtype))

            with tf.name_scope('train'):
                train_step = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(error)

            init = tf.global_variables_initializer()

        self.init = init
        self.inputs = inputs
        self.targets = targets

        self.error = error
        self.accuracy = accuracy
        self.is_training = is_training
        self.train_step = train_step
        self.outputs = logits

        # self.keep_prob_input = keep_prob_input
        # self.keep_prob_hidden = keep_prob_hidden

        return graph

    @staticmethod
    def joinTensors(left_tensor, right_tensor, name):
        return tf.concat((left_tensor, right_tensor), axis=1, name=name)

    @staticmethod
    def cutTensorInHalf(arra):
        feature_len = int(arra.get_shape()[1])
        half_len = feature_len // 2
        assert half_len == feature_len / 2  # only even number are accepted
        arl = tf.slice(arra, [0, 0], [-1, half_len])
        arr = tf.slice(arra, [0, half_len], [-1, -1])
        return arl, arr
