import tensorflow as tf


def tf_mse(outputs, targets):
    # return tf.reduce_mean(tf.square(tf.sub(targets, outputs)))
    return tf.reduce_mean(tf.squared_difference(targets, outputs))
