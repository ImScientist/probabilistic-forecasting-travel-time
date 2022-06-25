import logging
import datetime as dt
from google.cloud import bigquery

logger = logging.getLogger(__name__)


def get_query(
        start_date: dt.date,
        end_date: dt.date,
        bucket: str
):
    """ Get data """

    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    blob_prefix = f'nyc_taxi_trips_data/{start_date}_{end_date}'

    uri = f'gs://{bucket}/{blob_prefix}'

    query = f"""
        EXPORT DATA OPTIONS(
          uri='{uri}/data*.parquet',
          format='PARQUET',
          overwrite=false,
          compression='GZIP'
        ) AS
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
        WHERE DATE(pickup_datetime)
          BETWEEN '{start_date}' AND '{end_date}'    
    """

    return query


def collect_data(
        start_date: dt.date,
        end_date: dt.date,
        project_id: str,
        bucket: str
):
    """ Collect NYC taxi trip data from BigQuery and store it as a .parquet
    file in GCP """

    bq_client = bigquery.Client(project=project_id, location='US')  # noqa

    query = get_query(start_date, end_date, bucket)

    logger.info(f"Execute query ... {query}")

    job_config = bigquery.QueryJobConfig(
        allow_large_results=True,
        use_legacy_sql=False)

    # API request
    query_job = bq_client.query(query, job_config=job_config)

    # wait for the job to complete
    query_job.result()

    billed_mb = query_job.total_bytes_billed / (2 ** 20)  # noqa
    logger.info(f"Billed MBs: {billed_mb}")
