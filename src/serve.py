"""
    TODO: WIP

    Serve model notes;
    TODO: to simplify the model serving we could just completely remove
      the dropoff_area and the pickup area from the model (we still have
      the dropoff (lat, lon) and the pickup (lat lon))

    load_dir
    ├── checkpoints
    ├── model
    ├── model_deterministic
    ├── model_mean_std
    └── cluster.joblib
"""

import os
import json
import requests
import tensorflow as tf
import tensorflow_probability as tfp

from src.models.model import load_model

tfkl = tf.keras.layers
tfkc = tf.keras.callbacks
tfd = tfp.distributions
tfb = tfp.bijectors

dtype = tf.float32


def prepare_model_deterministic():
    """ Modify the model that outputs samples from a parametrized distribution
    to a model that outputs the parameters (loc, sig) of the distribution

    export LOAD_DIR=/home/ai/projects/carrrrs_ny/saved_models/ex_20

    docker run -it --rm -p 8501:8501 \
        -v "${LOAD_DIR}/model_deterministic":/models/model_deterministic/1" \
        -e MODEL_NAME=model_deterministic \
        tensorflow/serving:latest
    """

    load_dir = '/home/ai/projects/carrrrs_ny/saved_models/ex_20'
    model_dir = os.path.join(load_dir, 'model')
    model_deterministic_dir = os.path.join(load_dir, 'model_deterministic')

    model, cluster = load_model(model_dir)

    model_deterministic = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.layers[-2].output])

    model_deterministic.save(model_deterministic_dir)


def prepare_model_mean_std():
    """ Modify the model that outputs samples from a parametrized distribution
    to a model that outputs the mean and standard deviation of the distribution

    TODO: server is throwing an error when accepting requests for this model

    export LOAD_DIR=/home/ai/projects/carrrrs_ny/saved_models/ex_20

    docker run -it --rm -p 8501:8501 \
        -v "${LOAD_DIR}/model_mean_std":/models/model_mean_std/1" \
        -e MODEL_NAME=model_mean_std \
        tensorflow/serving:latest
    """

    load_dir = '/home/ai/projects/carrrrs_ny/saved_models/ex_20'
    model_dir = os.path.join(load_dir, 'model')
    model_mean_std_dir = os.path.join(load_dir, 'model_mean_std')

    model, cluster = load_model(model_dir)

    model_deterministic = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.layers[-2].output])

    inputs = model.inputs

    x = model_deterministic(inputs)

    two = tf.constant(2, dtype=dtype)
    one = tf.constant(2, dtype=dtype)
    half = tf.constant(0.5, dtype=dtype)

    mean = tfkl.Lambda(
        lambda t: tf.exp(t[..., :1] + half * tf.pow(t[..., 1:], two))
    )(x)

    stddev = tfkl.Lambda(
        lambda t: (
                (tf.exp(tf.pow(t[..., 1:], two)) - one) *
                tf.exp(tf.pow(t[..., 1:], two) + two * t[..., :1])
        )
    )(x)

    # mean = tfkl.Lambda(
    #     lambda t: tfd.LogNormal(loc=t[..., :1], scale=t[..., 1:]).mean()
    # )(x)
    #
    # stddev = tfkl.Lambda(
    #     lambda t: tfd.LogNormal(loc=t[..., :1], scale=t[..., 1:]).stddev()
    # )(x)

    output = tfkl.Concatenate(axis=-1)([mean, stddev])

    model_mean_std = tf.keras.Model(
        inputs=inputs,
        outputs=[output])

    model_mean_std.save(model_mean_std_dir)


def test_model():
    """ Test the previously prepared model

    """

    x0 = dict(
        dropoff_area=[1],
        dropoff_latitude=[40.734538],
        dropoff_longitude=[-73.991142],
        month=[1],
        passenger_count=[1],
        pickup_area=[2],
        pickup_latitude=[40.736072],
        pickup_longitude=[-73.991791],
        time=[571.0],
        trip_distance=[1.1],
        vendor_id=['1'],
        weekday=[1])

    data = json.dumps({
        "signature_name": "serving_default",
        "instances": [x0, x0, x0]
    })

    headers = {"content-type": "application/json"}

    json_response = requests.post(
        url='http://localhost:8501/v1/models/model_deterministic:predict',
        data=data,
        headers=headers)

    result = json_response.json()

    print(json_response.status_code == 200)
    print(result)
