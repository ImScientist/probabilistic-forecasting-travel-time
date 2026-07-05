from __future__ import annotations

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


def _read_pq_files(data_dir: str, columns: list[str], max_files: int = None) -> pd.DataFrame:
    """ Read and concatenate the parquet files in `data_dir` """

    files = sorted(glob.glob(f'{data_dir}/*.parquet'))
    files = files[:max_files]

    return pd.concat(
        (pd.read_parquet(f, columns=columns) for f in files),
        ignore_index=True)


def compute_feature_stats(data_dir: str, max_files: int = None) -> dict:
    """ Compute stats needed to build the model's preprocessing layers
    without an `.adapt()` pass over the dataset, by scanning the
    preprocessed parquet files in `data_dir`.

    Parameters
    ----------
    data_dir:
    max_files: take all files if max_files=None

    Returns
    -------
    {column: {'mean': ..., 'variance': ...}} for every numeric feature and
    {column: {'vocabulary': [...]}} for every categorical (integer) feature
    """

    columns = [*FLOAT_COLUMNS, *INT_COLUMNS]

    df = _read_pq_files(data_dir, columns, max_files)

    features = {
        col: df[col].to_numpy(dtype=np.int32 if col in INT_COLUMNS else np.float32)
        for col in columns}

    feature_stats = {
        col: {'mean': float(features[col].mean()), 'variance': float(features[col].var())}
        for col in FLOAT_COLUMNS}
    feature_stats.update({
        col: {'vocabulary': sorted(int(v) for v in np.unique(features[col]))}
        for col in INT_COLUMNS})

    return feature_stats


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
    """

    columns = [*FLOAT_COLUMNS, *INT_COLUMNS, 'target']

    df = _read_pq_files(data_dir, columns, max_files)

    features = {
        col: df[col].to_numpy(dtype=np.int32 if col in INT_COLUMNS else np.float32)
        for col in columns if col != 'target'}
    target = df['target'].to_numpy(dtype=np.float32)

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

    return ds
