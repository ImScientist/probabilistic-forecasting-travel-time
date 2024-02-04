"""
    TODO: WIP
    Serve the model

    load_dir
    ├── checkpoints
    ├── model
    ├── model_deterministic   #
    ├── model_mean_std        #
    └── model_attributes.joblib
"""

import os
import json
import logging
import requests
import tensorflow as tf
import tensorflow_probability as tfp

from model.models import ModelPDF

tfkl = tf.keras.layers
tfkc = tf.keras.callbacks
tfd = tfp.distributions
tfb = tfp.bijectors

dtype = tf.float32

logger = logging.getLogger(__name__)


def prepare_model_mean_std(load_dir: str):
    """ Modify the model that outputs samples from a parametrized distribution
    to a model that outputs the mean and std of the distribution

    Relevant only for elements of ModelPDF class
    TODO: server is throwing an error when accepting requests for ModelPDF
    """

    model_deterministic_dir = os.path.join(load_dir, 'model_mean_std')

    mdl = ModelPDF(load_dir=load_dir)

    model_deterministic = tf.keras.Model(
        inputs=mdl.model.inputs,
        outputs=[mdl.model.layers[-2].output])

    if mdl.distribution == 'lognormal':

        inputs = model_deterministic.inputs

        x = model_deterministic(inputs)

        mean = tfkl.Lambda(
            lambda t: tfd.LogNormal(
                loc=t[..., :1],
                scale=t[..., 1:]
            ).mean()
        )(x)

        stddev = tfkl.Lambda(
            lambda t: tfd.LogNormal(
                loc=t[..., :1],
                scale=t[..., 1:]
            ).stddev()
        )(x)

        output = tfkl.Concatenate(axis=-1)([mean, stddev])

        model_mean_std = tf.keras.Model(
            inputs=inputs,
            outputs=[output])

        model_mean_std.save(model_deterministic_dir)

    else:
        model_deterministic.save(model_deterministic_dir)


def test_model():
    """ Test the previously prepared model """

    x0 = {
        'dropoff_area': [0.000079],
        'dropoff_lat': [40.723752],
        'dropoff_lon': [-73.976968],
        'month': [1],
        'passenger_count': [1],
        'pickup_area': [0.000422],
        'pickup_lat': [40.744235],
        'pickup_lon': [-73.906306],
        'time': [800],
        'trip_distance': [2.3],
        'vendor_id': [0],
        'weekday': [3]}

    data = json.dumps({
        "signature_name": "serving_default",
        "instances": [x0, x0, x0]})

    headers = {"content-type": "application/json"}

    json_response = requests.post(
        url='http://localhost:8501/v1/models/model_mean_std:predict',
        data=data,
        headers=headers)

    result = json_response.json()

    logger.info(json_response.status_code == 200)
    logger.info(f'prediction: {result}')
