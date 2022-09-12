import io
import os
import logging
import tempfile
import functools
import numpy as np
import pandas as pd
import scipy.stats as sc
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Union
from PIL import Image

tfkl = tf.keras.layers
tfkc = tf.keras.callbacks
tfd = tfp.distributions
tfb = tfp.bijectors

dtype = tf.float32

logger = logging.getLogger(__name__)

# Check if the host and my machine are synchronized
CONTROL_VAR = 123


def lognormal_pdf(loc, scale):
    """ Lognormal definition as in Wikipedia (FU scipy) """

    def pdf_fn(x):
        norm = np.sqrt(2 * np.pi) * scale * x
        arg = - (np.log(x) - loc) ** 2 / (2 * scale ** 2)
        return np.exp(arg) / norm

    return pdf_fn


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


def plot_to_image(figure):
    """
        Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this
        call.
    """

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    figure.savefig(buf, format='png')

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Add the batch dimension
    image = tf.expand_dims(image, 0)

    return image


def store_model_architecture(model, log_dir: str):
    """ Store a non-interactive readable model architecture """

    with tempfile.NamedTemporaryFile('w', suffix=".png") as temp:
        _ = tf.keras.utils.plot_model(
            model,
            to_file=temp.name,
            show_shapes=True,
            dpi=64)

        im_frame = Image.open(temp.name)
        im_frame = np.asarray(im_frame)

        """ Log the figure """
        save_dir = os.path.join(log_dir, 'train')
        file_writer = tf.summary.create_file_writer(save_dir)

        with file_writer.as_default():
            tf.summary.image("model summary",
                             tf.constant(im_frame, dtype=tf.uint8)[tf.newaxis, ...],
                             step=0)


def fig_clustered_predictions(
        df: pd.DataFrame,
        clusters: int = 20,
        distribution_name: str = 'normal'
):
    """
        Plot distributions of observations with the same
        predicted distribution params p1, p2

    Params
    ------
      df: table with the columns p1, p2, p_cluster
    """

    # cluster distributions based on the similarity of their trainable params
    clustering_pipe = Pipeline([
        ('scaling', StandardScaler()),
        ('clustering', MiniBatchKMeans(n_clusters=clusters))])

    df['p_cluster'] = clustering_pipe.fit_predict(df[['p1', 'p2']].values)

    # relative number of elements per cluster
    sizes = (df
             .groupby(['p_cluster'])
             .agg(n=('p1', 'count'))
             .assign(n=lambda x: x / df.shape[0]))

    rows = clusters // 2
    cols = 2

    fig = plt.figure(figsize=(10 * cols, 3 * rows))

    for c in range(clusters):
        ax = plt.subplot(rows, cols, c + 1)

        frac = sizes.loc[c, 'n']

        cond = lambda x: x['p_cluster'] == c

        ax.hist(df.loc[cond, 'y'], bins=60, density=True,
                label=f'param-cluster {c:02d}; samples fraction: {frac:.3f}')

        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 1_000)

        p1_mean = df.loc[cond, 'p1'].mean()
        p2_mean = df.loc[cond, 'p2'].mean()

        if distribution_name == 'normal':
            pdf = sc.norm(loc=p1_mean, scale=p2_mean).pdf
        elif distribution_name == 'lognormal':
            pdf = lognormal_pdf(loc=p1_mean, scale=p2_mean)
        else:
            pdf = sc.gamma(a=p1_mean, scale=1 / p2_mean).pdf

        ax.plot(x, pdf(x), 'k-', lw=2, label='predicted pdf')

        ax.legend()

    return fig


def fig_mean_to_std_distribution(df: pd.DataFrame):
    """
        Plot the distribution of the mean to standard deviation ratios
        of the predicted distributions
    """

    label = 'mean to std ratio of predicted distributions'

    fig = plt.figure()
    plt.hist(df['mean'] / df['std'], bins=100, label=label)
    plt.legend()

    return fig


def fig_median_to_pct_range_distribution(
        df: pd.DataFrame, qtile_range: tuple[int, int]
):
    """
        Plot the distribution of the ratio btw the median and some percentile
        range.
    """

    label = 'mean to std ratio of predicted distributions'
    q1, q2 = qtile_range
    c1, c2 = f'q_{int(q1 * 100):03d}', f'q_{int(q2 * 100):03d}'

    fig = plt.figure()
    plt.hist(df['q_050'] / (df[c2] - df[c1]), bins=100, label=label)
    plt.legend()

    return fig


def fig_pct_skew(df: pd.DataFrame):
    """
        Plot distribution percentile of true values vs fraction of
        observations that belong to a lower percentile

    Params
    ------
      df: table with the columns:
         `pct`: percentile to which the observed value corresponds to
         `frac`: fraction of observations that belong to a lower predicted
              percentile
    """

    label = ('Predicted percentile vs fraction of observations '
             'that belong to a lower predicted percentile')

    fig = plt.figure(figsize=(15, 10))
    plt.scatter(df['pct'], df['frac'], alpha=0.2, s=20, label=label)
    plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')
    plt.legend()

    return fig


def fig_pct_skew_discrete(df: pd.DataFrame, quantiles: list):
    """
        For a discrete number of predicted percentiles calculate and visualize
        the fraction of observations that are below this percentile

    Params
    ------
      df: table with the columns:
         q_< pct >: predicted distribution percentiles
         y: observed value
    """

    frac = [(df['y'] < df[f'q_{int(q * 100):03d}']).mean()
            for q in quantiles]

    label = ('Predicted percentile vs fraction of observations \n'
             'that belong to a lower predicted percentile')

    fig = plt.figure()
    plt.scatter(quantiles, frac, label=label)
    plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')
    plt.legend()

    return fig


def store_model_predictions(model, ds, log_dir: str, clusters: int = 20):
    """ Compare predicted distributions against observations """

    save_dir = os.path.join(log_dir, 'train')
    file_writer = tf.summary.create_file_writer(save_dir)

    distribution_name = model.layers[-1].name
    distribution_fn = model.layers[-1].function

    model_deterministic = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.layers[-2].output])

    df = pd.DataFrame()

    df['y'] = np.hstack(list(ds.map(lambda x, y: y).as_numpy_iterator()))

    # predict distribution parameters
    df[['p1', 'p2']] = model_deterministic.predict(ds.map(lambda x, y: x))

    distribution = distribution_fn(df[['p1', 'p2']].values)[0]

    df['mean'] = distribution.mean()
    df['std'] = distribution.stddev()
    df['pct'] = distribution.cdf(df[['y']].values)

    # fraction of prediction with lower percentile
    df = df.assign(frac=lambda x: (x['pct'].sort_values() * 0 + 1).cumsum() / x.shape[0])

    """
        Figures
    """

    fig = fig_clustered_predictions(
        df=df, clusters=clusters, distribution_name=distribution_name)

    with file_writer.as_default():
        name = "predicted distributions"
        img = plot_to_image(fig)
        tf.summary.image(name, img, step=0)

    fig = fig_mean_to_std_distribution(df=df)

    with file_writer.as_default():
        name = "mean to std ratio of predicted distributions"
        img = plot_to_image(fig)
        tf.summary.image(name, img, step=0)

    fig = fig_pct_skew(df.iloc[:1_000])

    with file_writer.as_default():
        name = 'Predicted pct vs frac of observations with lower predicted pct'
        img = plot_to_image(fig)
        tf.summary.image(name, img, step=0)


# TODO: modify for the quantile loss fn
def train_evaluate(
        model,
        ds_tr,
        ds_va,
        log_dir: str,
        save_dir: str = None,
        epochs: int = 100,
        early_stopping_patience: int = 250,
        reduce_lr_patience: int = 100,
        histogram_freq: int = 0,
        profile_batch: tuple = (10, 15),
        verbose: int = 0,
        evaluate: bool = True
):
    callbacks = [
        tfkc.TensorBoard(log_dir=log_dir,
                         histogram_freq=histogram_freq,
                         profile_batch=profile_batch)
    ]

    if reduce_lr_patience is not None:
        callbacks.append(
            tfkc.ReduceLROnPlateau(factor=0.2,
                                   patience=reduce_lr_patience,
                                   verbose=verbose))

    if early_stopping_patience is not None:
        callbacks.append(
            tfkc.EarlyStopping(patience=early_stopping_patience))

    if save_dir:
        path = os.path.join(save_dir, 'checkpoints', 'cp.ckpt')
        callbacks.append(
            tfkc.ModelCheckpoint(path,
                                 save_weights_only=True,
                                 save_best_only=True))

    store_model_architecture(model, log_dir)

    _ = model.fit(
        ds_tr,
        validation_data=ds_va,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose)

    # Load the best model weights and save the full model
    if save_dir:
        checkpoint_path = os.path.join(save_dir, 'checkpoints', 'cp.ckpt')
        model.load_weights(checkpoint_path)
        model.save(save_dir, save_traces=False)

    model.evaluate(ds_tr)
    model.evaluate(ds_va)

    if evaluate:
        store_model_predictions(model, ds_va.take(1), log_dir)


def load_model(model_dir: str):
    """ Load model

    We have used `compile=False` to save the model OR something else?
    """

    model = tf.keras.models.load_model(
        model_dir,
        custom_objects={'neg_log_likelihood': neg_log_likelihood})

    return model
