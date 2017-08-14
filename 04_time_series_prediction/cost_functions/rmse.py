import tensorflow as tf

from cost_functions.mse import tf_mse


def tf_rmse(outputs, targets):
    return tf.sqrt(tf_mse(outputs, targets))
