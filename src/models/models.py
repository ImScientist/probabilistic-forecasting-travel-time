import os
import json
import joblib
import logging

import tensorflow as tf
import tensorflow_probability as tfp

from src.models import layer_generator
from src.models import losses

tfkl = tf.keras.layers
tfd = tfp.distributions
tfb = tfp.bijectors

dtype = tf.float32

logger = logging.getLogger(__name__)


def get_model_parametrized_dist(
        ds: tf.data.Dataset,
        num_feats: list[str],
        cat_int_feats: list[str],
        cat_str_feats: list[str],
        layer_sizes: tuple = (32, 32, 8),
        l2: float = 0.001,
        dropout: float = 0,
        dropout_min_layer_size: int = 12,
        batch_normalization: bool = False,
        distribution: str = 'normal'
) -> tf.keras.Model:
    """ Construct a model where the second to last layer is used to
    parametrize a Normal/LogNormal distribution
    """

    assert distribution in ('normal', 'lognormal')

    all_inputs, encoded_features = layer_generator.model_input_layer(
        ds=ds,
        num_feats=num_feats,
        cat_int_feats=cat_int_feats,
        cat_str_feats=cat_str_feats)

    x = tf.keras.layers.concatenate(encoded_features)

    for i, layer_size in enumerate(layer_sizes):
        x = layer_generator.composite_layer(
            x,
            layer_sizes=layer_size,
            l2=l2,
            dropout=dropout,
            dropout_min_layer_size=dropout_min_layer_size,
            batch_normalization=batch_normalization,
            name=f'cl_{i}')

    # second to last layer: no regularization, no dropout
    x1 = tfkl.Dense(1)(x)
    x2 = tfkl.Dense(1, activation='softplus')(x)
    x2 = tfkl.Lambda(lambda t: t + tf.constant(1e-3, dtype=dtype))(x2)
    x = tfkl.Concatenate(axis=-1)([x1, x2])

    dist = tfd.Normal if distribution == 'normal' else tfd.LogNormal

    output = tfp.layers.DistributionLambda(
        lambda t: dist(loc=t[..., :1], scale=t[..., 1:]),
        name=distribution
    )(x)

    model = tf.keras.Model(all_inputs, output)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.01),
        loss=losses.neg_log_likelihood)

    return model


def get_model_iqf(
        ds: tf.data.Dataset,
        num_feats: list[str],
        cat_int_feats: list[str],
        cat_str_feats: list[str],
        layer_sizes: tuple = (32, 32, 8),
        l2: float = 0.001,
        dropout: float = 0,
        dropout_min_layer_size: int = 12,
        batch_normalization: bool = False,
        quantiles: tuple = (.1, .3, .5, .7, .9)
):
    """ Model to predict multiple quantiles/percentiles that prevents the
    quantile crossing
    """

    n_quantiles = len(quantiles)
    loss_fn = losses.pinball_loss(quantiles=quantiles)

    all_inputs, encoded_features = layer_generator.model_input_layer(
        ds,
        num_feats=num_feats,
        cat_int_feats=cat_int_feats,
        cat_str_feats=cat_str_feats)

    x = tf.keras.layers.concatenate(encoded_features)

    for i, layer_size in enumerate(layer_sizes):
        x = layer_generator.composite_layer(
            x,
            layer_sizes=layer_size,
            l2=l2,
            dropout=dropout,
            dropout_min_layer_size=dropout_min_layer_size,
            batch_normalization=batch_normalization,
            name=f'cl_{i}')

    x1 = tfkl.Dense(1)(x)
    x2 = tfkl.Dense(n_quantiles - 1, activation='softplus')(x)
    x = tfkl.Concatenate(axis=-1)([x1, x2])

    # monotonically increasing outputs
    output = tfkl.Lambda(lambda y: tf.cumsum(y, axis=-1))(x)

    model = tf.keras.Model(all_inputs, output)

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.01),
        loss=loss_fn)

    return model


def load_model(load_dir: str, load_area_cluster: bool = True):
    """ Load model

    We have used `compile=False` to save the model
    """

    model_dir = os.path.join(load_dir, 'model')
    cluster_path = os.path.join(load_dir, 'cluster.joblib')

    if load_area_cluster:
        cluster = joblib.load(cluster_path)
    else:
        cluster = None

    model = tf.keras.models.load_model(
        model_dir,
        custom_objects={'neg_log_likelihood': losses.neg_log_likelihood})

    return model, cluster


def load_model_iqf(load_dir: str, load_area_cluster: bool = True):
    """ Load a percentile model """

    model_dir = os.path.join(load_dir, 'model')
    cluster_path = os.path.join(load_dir, 'cluster.joblib')
    quantiles_path = os.path.join(load_dir, 'quantiles.json')
    if load_area_cluster:
        cluster = joblib.load(cluster_path)
    else:
        cluster = None

    with open(quantiles_path, 'r') as f:
        quantiles = json.load(f)

    loss_fn = losses.pinball_loss(quantiles=quantiles)

    model = tf.keras.models.load_model(
        model_dir,
        custom_objects={'loss_fn': loss_fn})

    return model, cluster, quantiles
