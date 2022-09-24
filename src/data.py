import os
import logging

from google.cloud import bigquery

logger = logging.getLogger(__name__)


def get_query(month: int):
    """ Get data """

    query = f"""
        SELECT
          vendor_id,
          pickup_datetime,
          dropoff_datetime,
          pickup_longitude,
          pickup_latitude,
          dropoff_longitude,
          dropoff_latitude,
          trip_distance,
          passenger_count,
          fare_amount,
        FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2016`
        WHERE EXTRACT(MONTH FROM pickup_datetime) = {month}
        ORDER BY pickup_datetime
    """

    return query


def get_data_bq(month: int, project_id: str):
    """ Get NYC taxi trip data from BigQuery """

    bq_client = bigquery.Client(project=project_id, location='US')  # noqa

    query = get_query(month)

    logger.info(f"Execute query ... {query}")

    job_config = bigquery.QueryJobConfig(
        allow_large_results=True,
        use_legacy_sql=False)

    # API request
    query_job = bq_client.query(query, job_config=job_config)

    df = query_job.to_dataframe()

    billed_mb = query_job.total_bytes_billed / (2 ** 20)  # noqa
    logger.info(f"Billed MBs: {billed_mb}")

    return df


def get_data_bq_all(project_id: str, save_dir: str):
    """ Get NYC taxi trip data from BigQuery for the entire 2016 and store it
    locally as .parquet files """

    os.makedirs(save_dir, exist_ok=True)

    for month in range(1, 13):
        df = get_data_bq(month=month, project_id=project_id)
        path = os.path.join(save_dir, f'data_2016_{month}.parquet')
        df.to_parquet(path)
