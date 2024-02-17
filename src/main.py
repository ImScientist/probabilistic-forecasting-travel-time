from __future__ import annotations

import os
import json
import click
import logging
import warnings

import train
import settings
import default_args
from data import data_collection, data_preprocessing
from serve import prepare_servable_mean_std
from model.models import ModelIQF, ModelPDF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

warnings.filterwarnings("ignore", category=UserWarning)

data_raw_dir = os.path.join(settings.DATA_DIR, 'raw')
data_preprocessed_dir = os.path.join(settings.DATA_DIR, 'preprocessed')


@click.group()
def cli():
    pass


@cli.command("collect-data")
@click.option('--year', default=2016, type=int, help='year')
def collect_data_fn(year):
    """ Get NYC taxi trip data from BigQuery for a particular year and store
    it locally (separate .parquet files for every month)
    """

    data_collection.collect_data(save_dir=data_raw_dir, year=year)


@cli.command("preprocess-data")
@click.option('--tr', default=.8, type=float, help='training fraction')
@click.option('--va', default=.1, type=float, help='validation fraction')
@click.option('--te', default=.1, type=float, help='test fraction')
def preprocess_data_fn(tr, va, te):
    """ Preprocess data and split it into a training, validation and test
    datasets """

    assert tr + va + te == 1

    data_preprocessing.preprocess_pq_files(
        source_dir=data_raw_dir,
        output_dir=data_preprocessed_dir,
        tr_va_te_frac=(tr, va, te))


@cli.command("train")
@click.option("--model_wrapper", default="ModelPDF", type=click.Choice(["ModelPDF", "ModelIQF"]),
              help="TF-model wrapper")
@click.option("--model_args", default='{}', type=str, help="json string of the model args")
@click.option("--ds_args", default='{}', type=str, help="json string of the dataset args")
@click.option("--callbacks_args", default='{}', type=str, help="json string of the callback args")
@click.option("--training_args", default='{}', type=str, help="json string of the training args")
def train_fn(model_wrapper, model_args, ds_args, callbacks_args, training_args):
    """ Train a model """

    model_args = {**default_args.model_args, **json.loads(model_args)}
    ds_args = {**default_args.ds_args, **json.loads(ds_args)}
    callbacks_args = {**default_args.callbacks_args, **json.loads(callbacks_args)}
    training_args = {**default_args.training_args, **json.loads(training_args)}

    logger.info(
        f'model_wrapper: {model_wrapper}\n'
        f'model_args: {json.dumps(model_args, indent=2)}\n'
        f'ds_args: {json.dumps(ds_args, indent=2)}\n'
        f'callbacks_args: {json.dumps(callbacks_args, indent=2)}\n'
        f'training_args: {json.dumps(training_args, indent=2)}\n')

    train.train(
        model_wrapper=ModelPDF if model_wrapper == 'ModelPDF' else ModelIQF,
        ds_args=ds_args,
        model_args=model_args,
        callbacks_args=callbacks_args,
        training_args=training_args,
        data_preprocessed_dir=data_preprocessed_dir)


@cli.command("prepare-servable")
@click.option("--load_dir", required=True, type=str,
              help="Location of the stored model")
def prepare_servable_fn(load_dir: str):
    """ Prepare a model servable """

    prepare_servable_mean_std(load_dir)


if __name__ == "__main__":
    """
    PYTHONPATH=$(pwd) python src/main.py --help
    
    PYTHONPATH=$(pwd) python src/main.py train \
        --model_wrapper=ModelPDF \
        --model_args='{"l2": 0.0001, "batch_normalization": false, "layer_sizes": [64, [64, 64], [64, 64], 32, 8]}' \
        --ds_args='{"max_files": 2}' \
        --callbacks_args='{"period": 100, "profile_batch": 0}' \
        --training_args='{"epochs": 3000}'
    """

    cli()
