import os
import glob
import joblib
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_probability as tfp

from pyarrow.parquet import ParquetDataset
from sklearn.cluster import KMeans
from abc import abstractmethod
from collections import OrderedDict

tfkl = tf.keras.layers
tfkc = tf.keras.callbacks
tfd = tfp.distributions
tfb = tfp.bijectors

dtype = tf.float32

logger = logging.getLogger(__name__)


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


def get_pq_files_and_sizes(data_dir: str, max_files: int = None):
    """ Get the full paths and sizes of all .parquet files in data_dir """

    pq_files = [x for x in os.listdir(data_dir) if x.endswith('.parquet')]
    pq_files = sorted(pq_files)[:max_files]
    pq_files = np.array(pq_files)

    pq_sizes = []
    for pq_file in pq_files:
        path = os.path.join(data_dir, pq_file)
        size = sum(p.count_rows()
                   for p in ParquetDataset(path, use_legacy_dataset=False).fragments)
        pq_sizes.append(size)

    pq_sizes = np.array(pq_sizes)

    return pq_files, pq_sizes


# TODO: create an abstract method in the base class!!!!!!!!!!
# TODO: remove
def ds_from_files(
        data_dir: str,
        tr_va_te_fraq: tuple[float, float, float] = (.8, .1, .1),
        batch_size: int = 2 ** 20,
        shuffle_buffer_size: int = 2 ** 20,
        shuffle_seed: int = 1,
        cycle_length: int = 2
):
    """ Construct train, validation and test datasets from the .parquet files
    in data_dir

    Load original parquet files, preprocess, and store results as parquet files

    Parameters
    ----------
      data_dir:
      tr_va_te_fraq: train, validation and test fractions
      batch_size:
      shuffle_buffer_size: shuffling buffer size; if = 0 then elements are not
          shuffled
      shuffle_seed:
      cycle_length: controls the number of concurrently processed files
    """

    shuffle_seed = tf.constant(shuffle_seed, dtype=tf.int32)

    tr_fraq, va_fraq, te_fraq = tr_va_te_fraq

    assert tr_fraq + va_fraq + te_fraq == 1

    pq_files_and_sizes = get_pq_files_and_sizes(data_dir)

    columns_schema = {
        'time': tf.TensorSpec(tf.TensorShape([]), tf.float32),
        'trip_distance': tf.TensorSpec(tf.TensorShape([]), tf.float32),
        'pickup_location_id': tf.TensorSpec(tf.TensorShape([]), tf.int32),
        'dropoff_location_id': tf.TensorSpec(tf.TensorShape([]), tf.int32),
        'passenger_count': tf.TensorSpec(tf.TensorShape([]), tf.int32),
        'vendor_id': tf.TensorSpec(tf.TensorShape([]), tf.int32),
        'weekday': tf.TensorSpec(tf.TensorShape([]), tf.int32),
        'month': tf.TensorSpec(tf.TensorShape([]), tf.int32),
        'target': tf.TensorSpec(tf.TensorShape([]), tf.float32)}

    def pq_dataset(
            path: str, take_size: int = 0, skip_size: int = 0
    ):
        """

        Parameters
        ----------
          path: paruqet files path
          take_size: size of the dataset
          skip_size: skip the first `skip_size` elements from the dataset
        """

        ds = tfio.IODataset.from_parquet(path, columns=columns_schema)

        if shuffle_buffer_size > 0:
            ds = ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)

        if skip_size > 0:
            ds = ds.skip(skip_size)

        if take_size > 0:
            ds = ds.take(take_size)

        return ds

    def make_dataset(take_fraq: float, skip_fraq: float):
        """ Make dataset

        Parameters
        ----------
          take_fraq: fraction of elements from the original dataset that will
              be used
          skip_fraq: fraction of elements in the beginning that will be skipped
        """

        ds = (
            tf.data.Dataset
            .from_tensor_slices(pq_files_and_sizes)
            .interleave(
                lambda x: pq_dataset(
                    path=x['path'],
                    take_size=int(take_fraq * x['size']),
                    skip_size=int(skip_fraq * x['size'])),
                num_parallel_calls=tf.data.AUTOTUNE,
                block_length=batch_size,  # interleave blocks of batch_size records from each file
                cycle_length=cycle_length)
            .batch(batch_size)
            .map(lambda x: OrderedDict({k: tf.expand_dims(v, -1) for k, v in x.items()})))

        return ds

    ds_tr = make_dataset(take_fraq=tr_fraq, skip_fraq=0)
    ds_va = make_dataset(take_fraq=va_fraq, skip_fraq=tr_fraq)
    ds_te = make_dataset(take_fraq=0, skip_fraq=tr_fraq + va_fraq)

    return ds_tr, ds_va, ds_te


class DatasetGenerator:

    def __init__(self, **kwargs):
        return

    @abstractmethod
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Generate features from a pd.Dataframe """

    def preprocess_pq_files(self, source_dir: str, output_dir):
        """ Preprocess all parquet files by applying to them all feature
        transformations and storing them in `output_dir`
        """

        os.makedirs(output_dir, exist_ok=True)

        files = glob.glob('*.parquet', root_dir=source_dir)

        for file in files:
            path_load = os.path.join(source_dir, file)
            path_save = os.path.join(output_dir, file)

            (pd
             .read_parquet(path_load)
             .pipe(self.generate_features)
             .to_parquet(path_save))

    # TODO: deprecate soon?
    def df_to_dataset(
            self,
            df: pd.DataFrame,
            shuffle_buffer_size: int = 0,
            batch_size: int = None,
            prefetch_size: int = None,
            cache: bool = False
    ):
        """ Map dataframe to dataset """

        df = self.generate_features(df)
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

    def _raw_pq_to_dataset(
            self,
            path: str,
            columns_schema: dict,
            shuffle_buffer_size: int = 0,
            shuffle_seed: int = 1,
            take_size: int = 0,
            skip_size: int = 0
    ):
        """ Map parquet file to dataset """

        shuffle_seed = tf.constant(shuffle_seed, dtype=tf.int32)

        ds = tfio.IODataset.from_parquet(path, columns=columns_schema)

        if shuffle_buffer_size > 0:
            ds = ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)

        if skip_size > 0:
            ds = ds.skip(skip_size)

        if take_size > 0:
            ds = ds.take(take_size)

        return ds

    def pq_to_dataset(
            self,
            data_dir: str,
            columns_schema: dict,
            shuffle_buffer_size: int,
            shuffle_seed: int,
            take_fraq: float,
            skip_fraq: float,
            batch_size: int,
            cycle_length: int,
            max_files: int = None
    ):
        """ Make dataset from a directory full with parquet files """

        # Get key-value pairs with key = file path and value = # rows
        files, sizes = get_pq_files_and_sizes(data_dir, max_files)

        pq_files_sizes = {
            'path': files,
            'size': sizes,
            'take_size': (sizes * take_fraq).astype('int32'),
            'skip_size': (sizes * skip_fraq).astype('int32')}

        ds = (
            tf.data.Dataset
            .from_tensor_slices(pq_files_sizes)
            .interleave(
                lambda x: self._raw_pq_to_dataset(
                    path=x['path'],
                    columns_schema=columns_schema,
                    shuffle_buffer_size=shuffle_buffer_size,
                    shuffle_seed=shuffle_seed,
                    take_size=x['take_size'],
                    skip_size=x['skip_size']),
                num_parallel_calls=tf.data.AUTOTUNE,
                block_length=batch_size,  # interleave blocks of batch_size records from each file
                cycle_length=cycle_length)
            .batch(batch_size)
            .map(lambda x: OrderedDict({k: tf.expand_dims(v, -1)
                                        for k, v in x.items()})))

        return ds

    def tr_va_te_datasets(
            self,
            data_dir: str,
            max_files: int = None,
            tr_va_te_fraq: tuple[float, float, float] = (.8, .1, .1),
            batch_size: int = 2 ** 20,
            shuffle_buffer_size: int = 2 ** 20,
            shuffle_seed: int = 1,
            cycle_length: int = 2
    ):
        """ Construct train, validation and test datasets from the .parquet files
        in data_dir """

        tr_fraq, va_fraq, te_fraq = tr_va_te_fraq

        assert tr_fraq + va_fraq + te_fraq == 1

        columns_schema = {
            'time': tf.TensorSpec(tf.TensorShape([]), tf.float32),
            'trip_distance': tf.TensorSpec(tf.TensorShape([]), tf.float32),
            'pickup_location_id': tf.TensorSpec(tf.TensorShape([]), tf.int32),
            'dropoff_location_id': tf.TensorSpec(tf.TensorShape([]), tf.int32),
            'passenger_count': tf.TensorSpec(tf.TensorShape([]), tf.int32),
            'vendor_id': tf.TensorSpec(tf.TensorShape([]), tf.int32),
            'weekday': tf.TensorSpec(tf.TensorShape([]), tf.int32),
            'month': tf.TensorSpec(tf.TensorShape([]), tf.int32),
            'target': tf.TensorSpec(tf.TensorShape([]), tf.float32)}

        common_args = {
            'data_dir': data_dir,
            'columns_schema': columns_schema,
            'shuffle_buffer_size': shuffle_buffer_size,
            'shuffle_seed': shuffle_seed,
            'batch_size': batch_size,
            'cycle_length': cycle_length,
            'max_files': max_files}

        ds_tr = self.pq_to_dataset(
            take_fraq=tr_fraq, skip_fraq=0, **common_args)
        ds_va = self.pq_to_dataset(
            take_fraq=va_fraq, skip_fraq=tr_fraq, **common_args)
        ds_te = self.pq_to_dataset(
            take_fraq=0, skip_fraq=tr_fraq + va_fraq, **common_args)

        return ds_tr, ds_va, ds_te

    def save(self, save_dir: str):
        return


class DSGRawLocation(DatasetGenerator):

    def __init__(self, load_dir: str = None, area_clusters: int = 20):
        super(DSGRawLocation).__init__()

        self.cluster = KMeans(n_clusters=area_clusters)

        if load_dir is not None:
            self._load(load_dir)

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Generate features from a pd.Dataframe """

        """ Generate features from a table that has exact pickup and dropoff
        coordinates """

        lon_min, lon_max = -74.02, -73.86
        lat_min, lat_max = 40.67, 40.84
        trip_distance_max = 30

        types_map = {
            'pickup_longitude': 'float32',
            'pickup_latitude': 'float32',
            'dropoff_longitude': 'float32',
            'dropoff_latitude': 'float32',
            'trip_distance': 'float32',
            'time': 'float32',
            'passenger_count': 'int32',
            'weekday': 'int32',
            'month': 'int32',
            'pickup_area': 'int32',
            'dropoff_area': 'int32',
            'target': 'float32'}

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
        fit_cluster(self.cluster, df)

        df['target'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds()

        df['time'] = df['pickup_datetime'].dt.hour * 60 + df['pickup_datetime'].dt.minute

        df['weekday'] = df['pickup_datetime'].dt.weekday

        df['month'] = df['pickup_datetime'].dt.month

        df['pickup_area'] = self.cluster.predict(df[['pickup_longitude',
                                                     'pickup_latitude']].values)

        df['dropoff_area'] = self.cluster.predict(df[['dropoff_longitude',
                                                      'dropoff_latitude']].values)

        df['passenger_count'] = np.where(df['passenger_count'] < 7, df['passenger_count'], 7)

        df.drop(['pickup_datetime', 'dropoff_datetime', 'fare_amount'], axis=1, inplace=True)

        df = df.dropna().astype(types_map, copy=False)

        return df

    def save(self, save_dir: str):
        cluster_path = os.path.join(save_dir, 'cluster.joblib')
        joblib.dump(self.cluster, cluster_path)

    def _load(self, load_dir: str):
        cluster_path = os.path.join(load_dir, 'cluster.joblib')
        self.cluster = joblib.load(cluster_path)


class DSGMaskedLocation(DatasetGenerator):

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Generate features from a table where the pickup and dropoff coordinates
        are replaced with region ids """

        types_map = {
            'time': 'float32',
            'trip_distance': 'float32',
            'pickup_location_id': 'int32',
            'dropoff_location_id': 'int32',
            'passenger_count': 'int32',
            'vendor_id': 'int32',
            'weekday': 'int32',
            'month': 'int32',
            'target': 'float32'}

        cond = lambda x: (
            (x['dropoff_datetime'] - x['pickup_datetime'])
            .dt.total_seconds().between(1, 6000))

        df['target'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds()

        df['time'] = df['pickup_datetime'].dt.hour * 60 + df['pickup_datetime'].dt.minute

        df['weekday'] = df['pickup_datetime'].dt.weekday

        df['month'] = df['pickup_datetime'].dt.month

        df['passenger_count'] = np.where(df['passenger_count'] < 7, df['passenger_count'], 7)

        df = (df
              .loc[cond, list(types_map)]
              .dropna()
              .astype(types_map, copy=False))

        return df
