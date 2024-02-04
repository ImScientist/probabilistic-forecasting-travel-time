import os
import glob
import logging

import tensorflow as tf
import tensorflow_io as tfio

logger = logging.getLogger(__name__)


def pq_to_dataset(
        data_dir: str,
        batch_size: int,
        prefetch_size: int,
        cache: bool = True,
        cycle_length: int = 2,
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
    cycle_length: simultaneously opened files?
    max_files: take all files if max_files=None
    take_size: take all elements of the dataset if take_size=-1
    """

    columns_schema = {
        'time': tf.TensorSpec(tf.TensorShape([])),
        'trip_distance': tf.TensorSpec(tf.TensorShape([])),

        'pickup_lon': tf.TensorSpec(tf.TensorShape([])),
        'pickup_lat': tf.TensorSpec(tf.TensorShape([])),
        'pickup_area': tf.TensorSpec(tf.TensorShape([])),

        'dropoff_lon': tf.TensorSpec(tf.TensorShape([])),
        'dropoff_lat': tf.TensorSpec(tf.TensorShape([])),
        'dropoff_area': tf.TensorSpec(tf.TensorShape([])),

        'passenger_count': tf.TensorSpec(tf.TensorShape([]), tf.int32),
        'vendor_id': tf.TensorSpec(tf.TensorShape([]), tf.int32),
        'weekday': tf.TensorSpec(tf.TensorShape([]), tf.int32),
        'month': tf.TensorSpec(tf.TensorShape([]), tf.int32),
        'target': tf.TensorSpec(tf.TensorShape([]))}

    # files = sorted(glob.glob('*.parquet', root_dir=data_dir))
    files = sorted(glob.glob(f'{data_dir}/*.parquet'))
    files = [os.path.join(data_dir, x) for x in files]
    files = files[:max_files]

    ds = (
        tf.data.Dataset
        .from_tensor_slices(files)
        .interleave(
            lambda f: tfio.IODataset.from_parquet(
                filename=f,
                columns=columns_schema),
            num_parallel_calls=tf.data.AUTOTUNE,
            block_length=batch_size,
            cycle_length=cycle_length)
        .take(take_size)
        .batch(batch_size)
        .map(lambda x: ({k: tf.expand_dims(v, -1)
                         for k, v in x.items() if k != 'target'},
                        x['target'])))

    if prefetch_size is not None:
        ds = ds.prefetch(prefetch_size)

    if cache:
        ds = ds.cache()

    return ds
