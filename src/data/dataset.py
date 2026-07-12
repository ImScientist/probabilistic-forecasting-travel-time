from __future__ import annotations

import glob
import logging

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf

logger = logging.getLogger(__name__)

INT_COLUMNS = ('passenger_count', 'vendor_id', 'weekday', 'month')
STR_COLUMNS = ()
FLOAT_COLUMNS = (
    'time', 'trip_distance',
    'pickup_lon', 'pickup_lat', 'pickup_area',
    'dropoff_lon', 'dropoff_lat', 'dropoff_area')


def _read_pq_files(data_dir: str, columns: list[str], max_files: int = None) -> pd.DataFrame:
    """ Read and concatenate the parquet files in `data_dir` """

    files = sorted(glob.glob(f'{data_dir}/*.parquet'))
    files = files[:max_files]

    return pq.read_table(files, columns=columns, use_threads=True).to_pandas()


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
    {column: {'vocabulary': [...]}} for every categorical (integer or string) feature
    """

    columns = [*FLOAT_COLUMNS, *INT_COLUMNS, *STR_COLUMNS]

    df = _read_pq_files(data_dir, columns, max_files)

    features = {
        col: df[col].to_numpy(
            dtype=np.int32 if col in INT_COLUMNS else np.float32 if col in FLOAT_COLUMNS else str)
        for col in columns}

    feature_stats = {
        col: {'mean': float(features[col].mean()), 'variance': float(features[col].var())}
        for col in FLOAT_COLUMNS}
    feature_stats.update({
        col: {'vocabulary': sorted(int(v) for v in np.unique(features[col]))}
        for col in INT_COLUMNS})
    feature_stats.update({
        col: {'vocabulary': sorted(str(v) for v in np.unique(features[col]))}
        for col in STR_COLUMNS})

    return feature_stats


def pq_to_dataset(
        data_dir: str,
        batch_size: int,
        prefetch_size: int,
        cache: bool = True,
        shuffle: bool = True,
        max_files: int = None,
        take_size: int = -1
):
    """ Make dataset from a directory with parquet files

    The whole (preprocessed) dataset fits in memory.

    Parameters
    ----------
    data_dir:
    batch_size:
    prefetch_size:
    cache:
    shuffle: permute the batch order at every epoch
    max_files: take all files if max_files=None
    take_size: take all rows of the dataset if take_size=-1

    Returns
    -------
    ds: the tf.data.Dataset
    """

    logger.info(f'Load dataset from {data_dir}')

    columns = [*FLOAT_COLUMNS, *INT_COLUMNS, 'target']

    df = _read_pq_files(data_dir, columns, max_files)

    features = {
        col: df[col].to_numpy(dtype=np.int32 if col in INT_COLUMNS else np.float32)
        for col in columns if col != 'target'}
    target = df['target'].to_numpy(dtype=np.float32)

    n_rows = target.shape[0] if take_size == -1 else min(take_size, target.shape[0])

    # Group the rows into full batches. Each feature batch has the trailing
    # unit axis the model expects, shape (batch_size, 1); the target batch
    # stays 1-D, shape (batch_size,), matching the row-sliced version.
    n_batches = n_rows // batch_size
    n_full = n_batches * batch_size

    features_b = {
        col: arr[:n_full].reshape(n_batches, batch_size, 1)
        for col, arr in features.items()}
    target_b = target[:n_full].reshape(n_batches, batch_size)

    ds = tf.data.Dataset.from_tensor_slices((features_b, target_b))

    # Append the remaining rows as a final, smaller batch.
    if n_rows > n_full:
        features_r = {
            col: arr[n_full:n_rows].reshape(1, n_rows - n_full, 1)
            for col, arr in features.items()}
        target_r = target[n_full:n_rows].reshape(1, n_rows - n_full)
        ds = ds.concatenate(
            tf.data.Dataset.from_tensor_slices((features_r, target_r)))

    if cache:
        ds = ds.cache()

    if shuffle:
        # buffer >= number of batches -> a full permutation of the batch order
        ds = ds.shuffle(n_batches + 1, reshuffle_each_iteration=True)

    if prefetch_size is not None:
        ds = ds.prefetch(prefetch_size)

    return ds
