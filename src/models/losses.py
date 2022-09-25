import logging

import tensorflow as tf
import tensorflow_addons as tfa

dtype = tf.float32

logger = logging.getLogger(__name__)

# Check if the host and my machine are synchronized
CONTROL_VAR = 123


def neg_log_likelihood(y, rv_y):
    """ negative log-likelihood """

    return -rv_y.log_prob(y)


def pinball_loss(quantiles: tuple = (.1, .3, .5, .7, .9)):
    tau = tf.constant(list(quantiles), dtype=dtype)

    def loss_fn(y, y_pred):
        return tfa.losses.pinball_loss(y, y_pred, tau=tau)

    return loss_fn
