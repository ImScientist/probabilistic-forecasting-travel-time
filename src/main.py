import click

from src import settings
from src.data import get_data_bq_all


@click.group()
def cli():
    pass


@cli.command()
def collect_data():
    """ Get NYC taxi trip data from BigQuery for the entire 2016 and store it
    locally as separate .parquet files for every month
    """

    get_data_bq_all(
        project_id=settings.PROJECT_ID,
        save_dir=settings.DATA_DIR)


if __name__ == "__main__":
    cli()
