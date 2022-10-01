import json
import click
import logging

import src.default_args as default_args
import src.settings as settings
from src.data import get_data_bq_all
from src.train import train as train_fn

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


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


@cli.command()
@click.option("--ds_generator", default="DSGMaskedLocation", type=click.Choice(["DSGRawLocation", "DSGMaskedLocation"]),
              help="Dataset generator: use raw or masked location feature")
@click.option("--model_wrapper", default="ModelPDF", type=click.Choice(["ModelPDF", "ModelIQF"]),
              help="TF-model wrapper")
@click.option("--model_args", default='{}', type=str, help="json string of the model args")
@click.option("--ds_args", default='{}', type=str, help="json string of the dataset args")
@click.option("--callbacks_args", default='{}', type=str, help="json string of the callback args")
@click.option("--training_args", default='{}', type=str, help="json string of the training args")
def train(ds_generator, model_wrapper, model_args, ds_args, callbacks_args, training_args):
    """ Train a model """

    model_args = {**default_args.model_args[ds_generator], **json.loads(model_args)}
    ds_args = {**default_args.ds_args, **json.loads(ds_args)}
    callbacks_args = {**default_args.callbacks_args, **json.loads(callbacks_args)}
    training_args = {**default_args.training_args, **json.loads(training_args)}

    logger.info(
        f'ds_generator: {ds_generator}\n'
        f'model_wrapper: {model_wrapper}\n'
        f'model_args: {model_args}\n'
        f'ds_args: {ds_args}\n'
        f'callbacks_args: {callbacks_args}\n'
        f'training_args: {training_args}\n')

    train_fn(
        dataset_generator=ds_generator,
        model_wrapper=model_wrapper,
        ds_args=ds_args,
        mdl_args=model_args,
        callbacks_args=callbacks_args,
        training_args=training_args)


if __name__ == "__main__":
    """ PYTHONPATH=$(pwd) python src/main.py """
    cli()
