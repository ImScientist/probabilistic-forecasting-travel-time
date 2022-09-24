import os
import json
import joblib
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from sklearn.cluster import KMeans
from typing import Union

tfkl = tf.keras.layers
tfkc = tf.keras.callbacks
tfd = tfp.distributions
tfb = tfp.bijectors

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


def fit_cluster(cluster: KMeans, df: pd.DataFrame, sample_size: int = 10_000):
    """ Fit a clustering module if not fitted """

    if getattr(cluster, 'cluster_centers_', None) is None:
        logger.info("Fit clustering model...")

        df_sample = df.sample(sample_size)

        locations = np.vstack([
            df_sample[['pickup_longitude', 'pickup_latitude']].values,
            df_sample[['dropoff_longitude', 'dropoff_latitude']].values
        ])

        cluster.fit(locations)

    else:
        logger.info("Clustering model already fitted")


def generate_features(df: pd.DataFrame, cluster: KMeans) -> pd.DataFrame:
    """ Generate features """

    lon_min, lon_max = -74.02, -73.86
    lat_min, lat_max = 40.67, 40.84
    trip_distance_max = 30

    types_map = {
        'pickup_longitude': 'float32',
        'pickup_latitude': 'float32',
        'dropoff_longitude': 'float32',
        'dropoff_latitude': 'float32',
        'trip_distance': 'float32',
        'time': 'float32',  # 'int32' (it will be scaled later)

        'passenger_count': 'int32',

        'weekday': 'int32',
        'month': 'int32',
        'pickup_area': 'int32',
        'dropoff_area': 'int32',

        'target': 'float32'
    }

    cond = lambda x: (
            x['pickup_longitude'].between(lon_min, lon_max) &
            x['dropoff_longitude'].between(lon_min, lon_max) &
            x['pickup_latitude'].between(lat_min, lat_max) &
            x['dropoff_latitude'].between(lat_min, lat_max) &
            x['trip_distance'].between(0, trip_distance_max) &
            (x['dropoff_datetime'] - x['pickup_datetime']).dt.total_seconds().between(1, 3600)
    )

    df = df.loc[cond].copy()

    # Fit clustering fn if not fitted
    fit_cluster(cluster, df)

    df['target'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds()

    df['time'] = df['pickup_datetime'].dt.hour * 60 + df['pickup_datetime'].dt.minute

    df['weekday'] = df['pickup_datetime'].dt.weekday

    df['month'] = df['pickup_datetime'].dt.month

    df['pickup_area'] = cluster.predict(df[['pickup_longitude',
                                            'pickup_latitude']].values)

    df['dropoff_area'] = cluster.predict(df[['dropoff_longitude',
                                             'dropoff_latitude']].values)

    df['passenger_count'] = np.where(df['passenger_count'] < 7, df['passenger_count'], 7)

    df.drop(['pickup_datetime', 'dropoff_datetime', 'fare_amount'], axis=1, inplace=True)

    df = df.dropna().astype(types_map, copy=False)

    return df


def df_to_dataset(
        df: pd.DataFrame,
        cluster: KMeans,
        shuffle_buffer_size: int = 0,
        batch_size: int = None,
        prefetch_size: int = None,
        cache: bool = False
) -> tf.data.Dataset:
    """ Dataset creation """

    df = generate_features(df, cluster=cluster)
    target = df.pop('target')

    df = {key: value.values[..., np.newaxis] for key, value in df.items()}
    ds = tf.data.Dataset.from_tensor_slices((df, target))

    if shuffle_buffer_size > 0:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    if batch_size is not None:
        ds = ds.batch(batch_size)

    if prefetch_size is not None:
        ds = ds.prefetch(prefetch_size)

    if cache:
        ds = ds.cache()

    return ds


def get_normalization_layer(name, ds):
    """ Create and adapt a normalization layer """

    normalizer = tfkl.Normalization(axis=None)

    feature_ds = ds.map(lambda x, y: x[name])

    normalizer.adapt(feature_ds)

    return normalizer


def get_category_encoding_layer(name, ds, dtype, max_tokens=None):
    """ Create and adapt an encoding layer """

    # Create a layer that turns strings/integer values into integer indices.
    if dtype == 'string':
        index = tfkl.StringLookup(max_tokens=max_tokens)
    else:
        index = tfkl.IntegerLookup(max_tokens=max_tokens)

    feature_ds = ds.map(lambda x, y: x[name])

    index.adapt(feature_ds)

    encoder = tfkl.CategoryEncoding(num_tokens=index.vocabulary_size())

    return lambda x: encoder(index(x))


def model_input_layer(ds):
    """ Input layer from dataset """

    num_feats = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                 'dropoff_latitude', 'trip_distance', 'time']

    cat_int_feats = ['weekday', 'month', 'pickup_area',
                     'dropoff_area', 'passenger_count']

    cat_str_feats = ['vendor_id']

    all_inputs = []
    encoded_features = []

    for feat in num_feats:
        normalization_layer = get_normalization_layer(feat, ds)

        numeric_col = tf.keras.Input(shape=(1,), name=feat, dtype='float32')
        encoded_numeric_col = normalization_layer(numeric_col)

        all_inputs.append(numeric_col)
        encoded_features.append(encoded_numeric_col)

    for feat in cat_int_feats:
        encoding_layer = get_category_encoding_layer(feat, ds, dtype='int32')

        cat_col = tf.keras.Input(shape=(1,), name=feat, dtype='int32')
        encoded_cat_col = encoding_layer(cat_col)

        all_inputs.append(cat_col)
        encoded_features.append(encoded_cat_col)

    for feat in cat_str_feats:
        encoding_layer = get_category_encoding_layer(feat, ds, dtype='string')

        cat_col = tf.keras.Input(shape=(1,), name=feat, dtype='string')
        encoded_cat_col = encoding_layer(cat_col)

        all_inputs.append(cat_col)
        encoded_features.append(encoded_cat_col)

    return all_inputs, encoded_features


def composite_layer(
        x,
        layer_sizes: Union[tuple, int],
        l2: float = .001,
        dropout: float = 0,
        dropout_min_layer_size: int = 12,
        batch_normalization: bool = False,
        name: str = None
):
    layer_args = dict(
        l2=l2,
        dropout=dropout,
        dropout_min_layer_size=dropout_min_layer_size,
        batch_normalization=batch_normalization,
        name=name
    )

    if type(layer_sizes) == tuple:
        x = composite_layer_with_shortcut(x, layer_sizes=layer_sizes, **layer_args)
    else:
        x = composite_layer_without_shortcut(x, layer_size=layer_sizes, **layer_args)

    return x


def composite_layer_with_shortcut(
        x,
        layer_sizes: tuple = (32, 32),
        l2: float = .001,
        dropout: float = 0,
        dropout_min_layer_size: int = 12,
        batch_normalization: bool = False,
        name: str = None
):
    """ A residual block

    TODO: could we initialize the residual layers weights to 0?
    """

    shortcut = x

    for i, layer_size in enumerate(layer_sizes):

        x = tfkl.Dense(
            units=layer_size,
            activation='softplus',
            kernel_regularizer=tf.keras.regularizers.l2(l2),
            name=name + f'_{i:d}_dens'
        )(x)

        if batch_normalization:
            tfkl.BatchNormalization(name=name + f'_{i}_bn')(x)

        if dropout > 0 and layer_size > dropout_min_layer_size:
            x = tfkl.Dropout(dropout, name=name + f'_{i}_drop')(x)

    x = tfkl.Add(name=name + '_add')([shortcut, x])

    return x


def composite_layer_without_shortcut(
        x,
        layer_size: int = 32,
        l2: float = .001,
        dropout: float = 0,
        dropout_min_layer_size: int = 12,
        batch_normalization: bool = False,
        name: str = None
):
    x = tfkl.Dense(
        units=layer_size,
        activation='softplus',
        kernel_regularizer=tf.keras.regularizers.l2(l2),
        name=name + '_dens'
    )(x)

    if batch_normalization:
        tfkl.BatchNormalization(name=name + '_')(x)

    if dropout > 0 and layer_size > dropout_min_layer_size:
        x = tfkl.Dropout(dropout, name=name)(x)

    return x


def get_model(
        ds: tf.data.Dataset,
        layer_sizes: tuple = (32, 32, 8),
        l2: float = 0.001,
        dropout: float = 0,
        dropout_min_layer_size: int = 12,
        batch_normalization: bool = False,
        distribution: str = 'normal'
) -> tf.keras.Model:
    """ Construct the model """

    assert distribution in ('normal', 'lognormal')

    all_inputs, encoded_features = model_input_layer(ds)

    x = tf.keras.layers.concatenate(encoded_features)

    for i, layer_size in enumerate(layer_sizes):
        x = composite_layer(
            x,
            layer_sizes=layer_size,
            l2=l2,
            dropout=dropout,
            dropout_min_layer_size=dropout_min_layer_size,
            batch_normalization=batch_normalization,
            name=f'cl_{i}')

    # last layer: no regularization, no dropout
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
        loss=neg_log_likelihood)

    return model


def get_model_iqf(
        ds: tf.data.Dataset,
        layer_sizes: tuple = (32, 32, 8),
        l2: float = 0.001,
        dropout: float = 0,
        dropout_min_layer_size: int = 12,
        batch_normalization: bool = False,
        quantiles: tuple = (.1, .3, .5, .7, .9)
):
    """ Model to predict multiple quantiles/percentiles

    Prevent the quantile crossing
    """

    n_quantiles = len(quantiles)
    # tau = tf.constant(quantiles, dtype=dtype)
    loss_fn = pinball_loss(quantiles=quantiles)

    all_inputs, encoded_features = model_input_layer(ds)

    x = tf.keras.layers.concatenate(encoded_features)

    for i, layer_size in enumerate(layer_sizes):
        x = composite_layer(
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
        loss=loss_fn
        # loss=functools.partial(tfa.losses.pinball_loss, tau=tau)
    )

    return model


def load_model(load_dir: str):
    """ Load model

    We have used `compile=False` to save the model
    """

    model_dir = os.path.join(load_dir, 'model')
    cluster_path = os.path.join(load_dir, 'cluster.joblib')

    cluster = joblib.load(cluster_path)

    model = tf.keras.models.load_model(
        model_dir,
        custom_objects={'neg_log_likelihood': neg_log_likelihood})

    return model, cluster


def load_model_iqf(load_dir: str):
    """ Load a percentile model """

    model_dir = os.path.join(load_dir, 'model')
    cluster_path = os.path.join(load_dir, 'cluster.joblib')
    quantiles_path = os.path.join(load_dir, 'quantiles.json')

    cluster = joblib.load(cluster_path)

    with open(quantiles_path, 'r') as f:
        quantiles = json.load(f)

    loss_fn = pinball_loss(quantiles=quantiles)

    model = tf.keras.models.load_model(
        model_dir,
        custom_objects={'loss_fn': loss_fn})

    return model, cluster, quantiles
