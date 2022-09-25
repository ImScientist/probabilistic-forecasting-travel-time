import logging
import tensorflow as tf
import tensorflow_probability as tfp

tfkl = tf.keras.layers
tfkc = tf.keras.callbacks
tfd = tfp.distributions
tfb = tfp.bijectors

dtype = tf.float32

logger = logging.getLogger(__name__)


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


def model_input_layer(
        ds,
        num_feats: list[str],
        cat_int_feats: list[str],
        cat_str_feats: list[str]
):
    """ Input layer from dataset """

    # num_feats = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
    #              'dropoff_latitude', 'trip_distance', 'time']
    #
    # cat_int_feats = ['weekday', 'month', 'pickup_area',
    #                  'dropoff_area', 'passenger_count']
    #
    # cat_str_feats = ['vendor_id']

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
        layer_sizes: tuple | int,
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
