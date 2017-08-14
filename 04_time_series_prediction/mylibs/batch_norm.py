# -*- coding: utf-8 -*-
"""BatchNormalization Helper"""

import tensorflow as tf


class BatchNormer(object):
    # outputDim = inputs.get_shape()[-1]
    # ok this is static
    #
    # pop_mean
    # pop_var
    # population mean and population variance. For example this will converge to the overall mean when we see all batches
    # So every separate input has a batch size 256 and feature len of only 1.
    # So we are going to take the inputs of t=0 for 256 instances and just take the average of them
    # The question is. Should this be different than the next input? This would mean that the inputs are uncorrelated which is not true
    # Especially now that we use a sliding window across our price history.
    # Therefore... we should have the SAME population mean and SAME population variance across all inputs of the training so
    # pop_mean and pop_var should be repeated across inputs
    #
    # beta_offset
    # scale_gamma
    # with the same logic as above we should repeat also repeat beta offset and scale gamma across inputs
    #
    # # given that on axis=0 is where the batches extend (we want mean and var for each attribute)
    # batch_mean, batch_var = tf.nn.moments(inputs, axes=[0])
    #
    # decay = 0.999  # use numbers closer to 1 if you have more data
    # mean_of_train = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
    # var_of_train = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
    #
    # with tf.control_dependencies([mean_of_train, var_of_train]):
    #     normalized = tf.nn.batch_normalization(inputs,
    #                                            tf.cond(is_training, lambda: batch_mean, lambda: pop_mean),
    #                                            tf.cond(is_training, lambda: batch_var, lambda: pop_var),
    #                                            beta_offset, scale_gamma, epsilon)
    # In the logic described above the three above graph statements should be called multiple times for each input
    epsilon = 1e-3

    def __init__(self, bnId, inputs_or_outputDim, bo=None, sg=None):
        super(BatchNormer, self).__init__()

        if isinstance(inputs_or_outputDim, int):
            outputDim = inputs_or_outputDim
        else:
            outputDim = inputs_or_outputDim.get_shape()[-1]

        self.pop_mean = tf.Variable(tf.zeros(outputDim), trainable=False, name='pm_{}'.format(bnId))
        self.pop_var = tf.Variable(tf.ones(outputDim), trainable=False, name='pv_{}'.format(bnId))

        self.beta_offset = tf.Variable(tf.zeros(outputDim) if bo is None else bo, name='bo_{}'.format(bnId))
        self.scale_gamma = tf.Variable(tf.ones(outputDim) if sg is None else sg, name='sg_{}'.format(bnId))

    def batch_norm_wrapper(self, inputs, is_training):
        # given that on axis=0 is where the batches extend (we want mean and var for each attribute)
        batch_mean, batch_var = tf.nn.moments(inputs, axes=[0])

        decay = 0.999  # use numbers closer to 1 if you have more data
        mean_of_train = tf.assign(self.pop_mean, self.pop_mean * decay + batch_mean * (1 - decay))
        var_of_train = tf.assign(self.pop_var, self.pop_var * decay + batch_var * (1 - decay))

        with tf.control_dependencies([mean_of_train, var_of_train]):
            normalized = tf.nn.batch_normalization(inputs,
                                                   tf.cond(is_training, lambda: batch_mean, lambda: self.pop_mean),
                                                   tf.cond(is_training, lambda: batch_var, lambda: self.pop_var),
                                                   self.beta_offset, self.scale_gamma, self.epsilon)

        return normalized


def batchNormWrapper(bnId, inputs, is_training, epsilon=1e-3, bo=None, sg=None):
    """byExponentialMovingAvg
    recall that axes below is defined to the first axis zero(0)
    The convention is that this axis is dedicated to express the different instances in the batch
    We are taking the mean and the variance over the batch"""

    outputDim = inputs.get_shape()[-1]

    pop_mean = tf.Variable(tf.zeros(outputDim), trainable=False, name='pm_{}'.format(bnId))
    pop_var = tf.Variable(tf.ones(outputDim), trainable=False, name='pv_{}'.format(bnId))

    beta_offset = tf.Variable(tf.zeros(outputDim) if bo is None else bo, name='bo_{}'.format(bnId))
    scale_gamma = tf.Variable(tf.ones(outputDim) if sg is None else sg, name='sg_{}'.format(bnId))

    # given that on axis=0 is where the batches extend (we want mean and var for each attribute)
    batch_mean, batch_var = tf.nn.moments(inputs, axes=[0])

    decay = 0.999  # use numbers closer to 1 if you have more data
    mean_of_train = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
    var_of_train = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

    with tf.control_dependencies([mean_of_train, var_of_train]):
        normalized = tf.nn.batch_normalization(inputs,
                                               tf.cond(is_training, lambda: batch_mean, lambda: pop_mean),
                                               tf.cond(is_training, lambda: batch_var, lambda: pop_var),
                                               beta_offset, scale_gamma, epsilon)

    return normalized


def fully_connected_layer_with_batch_norm(fcId, inputs, input_dim, output_dim, is_training, nonlinearity=tf.nn.relu,
                                          avoidDeadNeurons=0.,
                                          w=None, b=None, bo=None, sg=None):
    weights = tf.Variable(
        tf.truncated_normal([input_dim, output_dim], stddev=2. / (input_dim + output_dim) ** 0.5) if w is None else w,
        name='weights_{}'.format(fcId)
    )

    biases = tf.Variable(avoidDeadNeurons * tf.ones([output_dim]) if b is None else b, name='biases_{}'.format(fcId))

    out_affine = tf.matmul(inputs, weights) + biases

    batchNorm = batchNormWrapper(fcId, out_affine, is_training, bo=bo, sg=sg)

    outputs = nonlinearity(batchNorm)
    return outputs


def fully_connected_layer_with_batch_norm_and_l2(fcId, inputs, input_dim, output_dim,
                                                 is_training, lamda2, nonlinearity=tf.nn.relu,
                                                 avoidDeadNeurons=0.,
                                                 w=None, b=None):
    """
    hiddenLayer, hiddenRegularizer = fully_connected_layer_with_batch_norm_and_l2(
        0, inputs_prob,
        inputDim, hidden_dim,
        nonlinearity=tf.nn.tanh,
        training=training,
        lamda2=lamda2
    )
    and also
    lamda2 * tf.nn.l2_loss(w1)
    """

    weights = tf.Variable(
        tf.truncated_normal([input_dim, output_dim], stddev=2. / (input_dim + output_dim) ** 0.5) if w is None else w,
        name='weights_{}'.format(fcId)
    )

    regularizer = lamda2 * tf.nn.l2_loss(weights)

    avoidDeadNeurons = 0.1 if nonlinearity == tf.nn.relu else avoidDeadNeurons  # prevent zero when relu

    biases = tf.Variable(avoidDeadNeurons * tf.ones([output_dim]) if b is None else b, name='biases_{}'.format(fcId))

    # out_affine = tf.matmul(inputs, weights) + biases
    out_affine = tf.add(tf.matmul(inputs, weights), biases)

    batchNorm = batchNormWrapper(fcId, out_affine, is_training)

    outputs = nonlinearity(batchNorm)

    return outputs, regularizer
