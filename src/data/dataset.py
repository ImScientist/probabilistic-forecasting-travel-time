import glob
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)

INT_COLUMNS = ('passenger_count', 'vendor_id', 'weekday', 'month')
FLOAT_COLUMNS = (
    'time', 'trip_distance',
    'pickup_lon', 'pickup_lat', 'pickup_area',
    'dropoff_lon', 'dropoff_lat', 'dropoff_area')


def pq_to_dataset(
        data_dir: str,
        batch_size: int,
        prefetch_size: int,
        cache: bool = True,
        max_files: int = None,
        take_size: int = -1
):
    """ Make dataset from a directory with parquet files

    Parameters
    ----------
    data_dir:
    batch_size:
    prefetch_size:
    cache:
    max_files: take all files if max_files=None
    take_size: take all elements of the dataset if take_size=-1

    Returns
    -------
    ds: the tf.data.Dataset
    feature_stats: {column: {'mean': ..., 'variance': ...}} for every
        numeric feature, computed directly from the loaded data
    """

    columns = [*FLOAT_COLUMNS, *INT_COLUMNS, 'target']

    files = sorted(glob.glob(f'{data_dir}/*.parquet'))
    files = files[:max_files]

    df = pd.concat(
        (pd.read_parquet(f, columns=columns) for f in files),
        ignore_index=True)

    features = {
        col: df[col].to_numpy(dtype=np.int32 if col in INT_COLUMNS else np.float32)
        for col in columns if col != 'target'}
    target = df['target'].to_numpy(dtype=np.float32)

    # Mean/variance of the numeric features, computed once while the data is
    # already in memory. This lets the model build its normalization layers
    # without an extra `.adapt()` pass over the whole dataset.
    feature_stats = {
        col: {'mean': float(features[col].mean()), 'variance': float(features[col].var())}
        for col in FLOAT_COLUMNS}

    ds = (
        tf.data.Dataset
        .from_tensor_slices((features, target))
        .take(take_size)
        .batch(batch_size)
        .map(lambda x, y: (
            {k: tf.expand_dims(v, -1) for k, v in x.items()}, y)))

    if prefetch_size is not None:
        ds = ds.prefetch(prefetch_size)

    if cache:
        ds = ds.cache()

    return ds, feature_stats
