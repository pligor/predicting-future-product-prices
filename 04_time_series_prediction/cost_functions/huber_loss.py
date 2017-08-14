import tensorflow as tf
import numpy as np


# def _huber_loss_np(y_true, y_pred):
#     err = y_true - y_pred
#     absolute = np.abs(err)
#     print absolute
#
#     ifthen = 0.5 * err
#     ifelse = absolute - 0.5
#
#     return np.where(absolute < 1.0, ifthen, ifelse)
def _huber_loss_np(y_true, y_pred, delta=1):
    """https://en.wikipedia.org/wiki/Huber_loss"""
    err = y_true - y_pred
    absolute = np.abs(err)
    # print absolute
    ifthen = 0.5 * (err ** 2)
    ifelse = delta * absolute - 0.5 * (delta ** 2)
    # print absolute < 1.0
    # print ifthen
    # print ifelse
    return np.where(absolute <= 1.0 * delta, ifthen, ifelse)


def _huber_loss_tf(y_true, y_pred):
    err = y_true - y_pred
    absolute = tf.abs(err)
    return tf.where(absolute < 1.0,
                    0.5 * tf.square(err),
                    absolute - 0.5)  # if, then, else


def huber_loss(y_true, y_pred):
    if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        return _huber_loss_np(y_true=y_true, y_pred=y_pred)
    else:
        return _huber_loss_tf(y_true=y_true, y_pred=y_pred)


def huberLoss(y_true, y_pred):
    return huber_loss(y_true=y_true, y_pred=y_pred)
