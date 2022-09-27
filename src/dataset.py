import os
import joblib
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.cluster import KMeans

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


def generate_features_01(df: pd.DataFrame, cluster: KMeans) -> pd.DataFrame:
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


def generate_features_02(df: pd.DataFrame) -> pd.DataFrame:
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


def df_to_dataset_01(
        df: pd.DataFrame,
        cluster: KMeans,
        shuffle_buffer_size: int = 0,
        batch_size: int = None,
        prefetch_size: int = None,
        cache: bool = False
) -> tf.data.Dataset:
    """ Dataset creation from a table that has exact pickup and dropoff
    coordinates """

    df = generate_features_01(df, cluster=cluster)
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


def df_to_dataset_02(
        df: pd.DataFrame,
        shuffle_buffer_size: int = 0,
        batch_size: int = None,
        prefetch_size: int = None,
        cache: bool = False
) -> tf.data.Dataset:
    """ Dataset creation from a table where the pickup and dropoff coordinates
    are replaced with region ids """

    df = generate_features_02(df)
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


class DatasetGenerator01:

    def __init__(
            self, load_dir: str = None, area_clusters: int = 20, **kwargs
    ):
        self.cluster = KMeans(n_clusters=area_clusters)

        if load_dir is not None:
            self._load(load_dir)

    def df_to_dataset(
            self,
            df: pd.DataFrame,
            shuffle_buffer_size: int = 0,
            batch_size: int = None,
            prefetch_size: int = None,
            cache: bool = False
    ):
        ds = df_to_dataset_01(
            df=df,
            cluster=self.cluster,
            shuffle_buffer_size=shuffle_buffer_size,
            batch_size=batch_size,
            prefetch_size=prefetch_size,
            cache=cache)

        return ds

    def save(self, save_dir: str):
        cluster_path = os.path.join(save_dir, 'cluster.joblib')
        joblib.dump(self.cluster, cluster_path)

    def _load(self, load_dir: str):
        cluster_path = os.path.join(load_dir, 'cluster.joblib')
        self.cluster = joblib.load(cluster_path)


class DatasetGenerator02:

    def __init__(self, **kwargs):
        return

    def df_to_dataset(
            self,
            df: pd.DataFrame,
            shuffle_buffer_size: int = 0,
            batch_size: int = None,
            prefetch_size: int = None,
            cache: bool = False
    ):
        ds = df_to_dataset_02(
            df=df,
            shuffle_buffer_size=shuffle_buffer_size,
            batch_size=batch_size,
            prefetch_size=prefetch_size,
            cache=cache)

        return ds

    def save(self, save_dir: str):
        return
