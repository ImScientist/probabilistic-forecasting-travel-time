import os
import logging
import pandas as pd
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)


def get_data(save_dir: str, year: int):
    """ Get NYC taxi data and taxi-zones data for a particular year """

    os.makedirs(save_dir, exist_ok=True)

    columns = [
        'VendorID',
        'tpep_pickup_datetime',
        'tpep_dropoff_datetime',
        'PULocationID',
        'DOLocationID',
        'trip_distance',
        'passenger_count']

    for month in range(1, 13):
        uri = ('https://d37ci6vzurychx.cloudfront.net/trip-data/'
               f'yellow_tripdata_{year}-{month:02d}.parquet')

        dst = os.path.join(save_dir, f'data_{year}-{month:02d}.parquet')

        logger.info(f'Store data in {dst}')
        pd.read_parquet(uri, columns=columns).to_parquet(dst)

    # Store taxi-zones data
    uri = 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip'
    dst = os.path.join(save_dir, 'taxi_zones.zip')

    urlretrieve(uri, dst)
