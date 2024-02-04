import os
import glob
import logging

import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def train_val_test_split(
        index: pd.Index,
        tr_va_te_frac: tuple[float, float, float],
        random_state: int = None
):
    """ Split data into training, validation and test 10% parts """

    frac_tr, frac_va, frac_te = tr_va_te_frac

    assert frac_tr + frac_va + frac_te == 1

    idx_tr, idx_rest = train_test_split(
        index,
        train_size=frac_tr,
        random_state=random_state)

    idx_va, idx_te = train_test_split(
        idx_rest,
        train_size=frac_va / (frac_va + frac_te),
        random_state=random_state)

    del idx_rest

    return idx_tr, idx_va, idx_te


def taxi_zones_summary(taxi_zones_path: str):
    """ Get a dataframe with center and area of each taxi zone """

    df = (
        gpd
        .read_file(taxi_zones_path)
        .to_crs("epsg:4326")
        .rename(columns={'LocationID': 'location_id'})
        .assign(lon=lambda x: x.geometry.centroid.x,
                lat=lambda x: x.geometry.centroid.y,
                area=lambda x: x.geometry.area)
        .loc[:, ['location_id', 'lon', 'lat', 'area']]
        .set_index('location_id'))

    return df


def generate_features(df: pd.DataFrame, taxi_zones: pd.DataFrame) -> pd.DataFrame:
    """ Generate features from a table where the pickup and dropoff coordinates
    are replaced with region ids """

    new_column_names = {
        'VendorID': 'vendor_id',
        'tpep_pickup_datetime': 'pickup_datetime',
        'tpep_dropoff_datetime': 'dropoff_datetime',
        'PULocationID': 'pickup_location_id',
        'DOLocationID': 'dropoff_location_id',
        'trip_distance': 'trip_distance',
        'passenger_count': 'passenger_count'}

    types_map = {
        'time': 'float32',
        'trip_distance': 'float32',
        'pickup_lon': 'float32',
        'pickup_lat': 'float32',
        'pickup_area': 'float32',
        'dropoff_lon': 'float32',
        'dropoff_lat': 'float32',
        'dropoff_area': 'float32',
        'passenger_count': 'int32',
        'vendor_id': 'int32',
        'weekday': 'int32',
        'month': 'int32',
        'target': 'float32'}

    cond = lambda x: (
        (x['dropoff_datetime'] - x['pickup_datetime'])
        .dt.total_seconds().between(1, 6_000))

    df = (
        df
        .rename(columns=new_column_names)
        .assign(target=lambda x: (x['dropoff_datetime'] - x['pickup_datetime']).dt.total_seconds(),
                time=lambda x: x['pickup_datetime'].dt.hour * 60 + x['pickup_datetime'].dt.minute,
                weekday=lambda x: x['pickup_datetime'].dt.weekday,
                month=lambda x: x['pickup_datetime'].dt.month,
                passenger_count=lambda x: np.where(x['passenger_count'] < 7, x['passenger_count'], 7))
        .merge(right=(taxi_zones
                      .rename(columns={'lon': 'pickup_lon',
                                       'lat': 'pickup_lat',
                                       'area': 'pickup_area'})),
               how='left',
               left_on='pickup_location_id',
               right_index=True)
        .merge(right=(taxi_zones
                      .rename(columns={'lon': 'dropoff_lon',
                                       'lat': 'dropoff_lat',
                                       'area': 'dropoff_area'})),
               how='left',
               left_on='dropoff_location_id',
               right_index=True)
        .loc[cond, list(types_map)]
        .dropna()
        .astype(types_map, copy=False))

    return df


def preprocess_pq_files(
        source_dir: str,
        output_dir: str,
        tr_va_te_frac: tuple[float, float, float] = (.8, .1, .1)
):
    """ Preprocess all parquet files by applying to them all feature
    transformations, splitting them into a train, validation and test datasets
    and storing them in `output_dir`

        `output_dir`
        ├── train/
        ├── validation/
        └── test/
    """

    os.makedirs(os.path.join(output_dir, 'train'))
    os.makedirs(os.path.join(output_dir, 'validation'))
    os.makedirs(os.path.join(output_dir, 'test'))

    taxi_zones_path = os.path.join(source_dir, 'taxi_zones.zip')
    taxi_zones = taxi_zones_summary(taxi_zones_path)

    files = glob.glob('*.parquet', root_dir=source_dir)
    files = sorted(files)

    for file in files:
        path_load = os.path.join(source_dir, file)

        logger.info(f'Preprocess {path_load}...')

        path_save_tr = os.path.join(output_dir, 'train', file)
        path_save_va = os.path.join(output_dir, 'validation', file)
        path_save_te = os.path.join(output_dir, 'test', file)

        df = pd.read_parquet(path_load)
        df = generate_features(df, taxi_zones=taxi_zones)

        idx_tr, idx_va, idx_te = train_val_test_split(
            index=df.index,
            tr_va_te_frac=tr_va_te_frac)

        df.loc[idx_tr].to_parquet(path_save_tr)
        df.loc[idx_va].to_parquet(path_save_va)
        df.loc[idx_te].to_parquet(path_save_te)
