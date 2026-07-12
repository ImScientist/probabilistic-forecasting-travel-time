from __future__ import annotations

import os
import json
import logging

import tensorflow as tf
import tensorflow_probability as tfp

from abc import abstractmethod
from model import layer_generator
from model.config import ModelConfig
from evaluate import (
    evaluate_parametrized_pdf_model,
    evaluate_percentile_model)

tfkl = tf.keras.layers
tfd = tfp.distributions
tfb = tfp.bijectors

dtype = tf.float32

logger = logging.getLogger(__name__)


def neg_log_likelihood(y, rv_y):
    """ negative log-likelihood """

    return -rv_y.log_prob(y)


def pinball_loss(quantiles: tuple = (.1, .3, .5, .7, .9)):
    """ Average pinball loss across the given quantiles """

    tau = tf.constant(list(quantiles), dtype=dtype)

    def loss_fn(y, y_pred):
        delta = tf.cast(y, dtype) - y_pred
        loss = tf.maximum(tau * delta, (tau - 1) * delta)
        return tf.reduce_mean(loss, axis=-1)

    return loss_fn


class ModelWrapper:
    def __init__(
            self,
            config: ModelConfig | dict = None,
            **kwargs
    ):
        # architecture config (accepts a ModelConfig or a plain dict from json
        # args; overwritten by `_load` when a stored model is loaded)
        self.config = ModelConfig.from_dict(config)

        # defined in the child class
        self.model = None
        self.custom_objects = None

    @abstractmethod
    def _init_model(self, ds, feature_stats: dict = None, optimizer=None):
        """ Initialize a model """

    def _load(self, load_dir: str):
        """ Load the model and all relevant class attributes """

        model_dir = os.path.join(load_dir, 'model')
        attributes_path = os.path.join(load_dir, 'model_attributes.json')
        feature_stats_path = os.path.join(load_dir, 'feature_stats.json')

        self.model = tf.keras.models.load_model(
            model_dir,
            custom_objects=self.custom_objects)

        # Load other attributes
        with open(attributes_path, 'r') as f:
            attributes = json.load(f)

        # The architecture config is stored as a nested "config" object (see
        # `save`); rebuild the dataclass and set everything else as attributes.
        self.config = ModelConfig.from_dict(attributes.pop('config', None))
        for key, value in attributes.items():
            setattr(self, key, value)

        # Load the normalization/vocabulary stats used to build the model's
        # preprocessing layers (see `feature_stats` in `data/dataset.py`)
        with open(feature_stats_path, 'r') as f:
            self.feature_stats = json.load(f)

    def save(self, save_dir: str):
        """ Save the model and other class attributes """

        model_dir = os.path.join(save_dir, 'model')
        attributes_path = os.path.join(save_dir, 'model_attributes.json')
        feature_stats_path = os.path.join(save_dir, 'feature_stats.json')

        attributes = {k: getattr(self, k) for k in vars(self).keys()
                      if k not in ('model', 'custom_objects', 'feature_stats', 'config')}
        # store the architecture config as a nested "config" object
        attributes['config'] = self.config.to_dict()

        self.model.save(model_dir)  # save_traces=False

        with open(attributes_path, 'w') as f:
            json.dump(attributes, f, indent='\t')

        with open(feature_stats_path, 'w') as f:
            json.dump(self.feature_stats, f, indent='\t')

    @abstractmethod
    def evaluate_model(self, ds, log_dir: str, log_data: dict = None):
        """ Evaluate model """


class ModelPDF(ModelWrapper):
    def __init__(
            self,
            config: ModelConfig | dict = None,
            distribution: str = 'lognormal',
            optimizer=None,
            ds=None,
            feature_stats: dict = None,
            load_dir: str = None,
            **kwargs
    ):

        super(ModelPDF, self).__init__(config=config)

        assert ds is not None or load_dir is not None
        assert distribution in ('normal', 'lognormal')

        self.distribution = distribution
        self.custom_objects = {
            'neg_log_likelihood': neg_log_likelihood}
        self.feature_stats = feature_stats

        if load_dir is not None:
            self._load(load_dir)
        else:
            self._init_model(ds, feature_stats, optimizer)

    def _init_model(self, ds, feature_stats: dict = None, optimizer=None):
        """ Initialize a model where the second to last layer is used to
        parametrize a Normal/LogNormal distribution
        """

        all_inputs, encoded_features = layer_generator.model_input_layer(
            ds=ds,
            features=self.config.features,
            embedding_dim=self.config.embedding_dim,
            feature_stats=feature_stats)

        x = tf.keras.layers.concatenate(encoded_features)

        for i, layer_size in enumerate(self.config.layer_sizes):
            x = layer_generator.composite_layer(
                x,
                layer_sizes=layer_size,
                l2=self.config.l2,
                dropout=self.config.dropout,
                dropout_min_layer_size=self.config.dropout_min_layer_size,
                batch_normalization=self.config.batch_normalization,
                name=f'cl_{i}')

        # second to last layer: no regularization, no dropout
        x1 = tfkl.Dense(1)(x)
        x2 = tfkl.Dense(1, activation='softplus')(x)
        x2 = tfkl.Lambda(lambda t: t + tf.constant(1e-3, dtype=dtype))(x2)
        x = tfkl.Concatenate(axis=-1)([x1, x2])

        dist = tfd.Normal if self.distribution == 'normal' else tfd.LogNormal

        output = tfp.layers.DistributionLambda(
            lambda t: dist(loc=t[..., :1], scale=t[..., 1:]),
            name=self.distribution
        )(x)

        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=0.001)

        self.model = tf.keras.Model(all_inputs, output)
        self.model.compile(
            optimizer=optimizer,
            loss=neg_log_likelihood)

    def evaluate_model(self, ds, log_dir: str, log_data: dict = None):
        """ Evaluate model """

        evaluate_parametrized_pdf_model(
            self.model, ds=ds, log_dir=log_dir, log_data=log_data)


class ModelIQF(ModelWrapper):
    def __init__(
            self,
            config: ModelConfig | dict = None,
            quantiles: tuple = (.05, .15, .3, .5, .7, .85, .95),
            quantile_range: tuple[float, float] = (.15, .85),
            optimizer=None,
            ds=None,
            feature_stats: dict = None,
            load_dir: str = None,
            **kwargs
    ):
        super(ModelIQF, self).__init__(config=config)

        assert ds is not None or load_dir is not None

        self.quantiles = quantiles
        self.quantile_range = quantile_range
        self.custom_objects = {
            'loss_fn': pinball_loss(quantiles=self.quantiles)}
        self.feature_stats = feature_stats

        if load_dir is not None:
            self._load(load_dir)
        else:
            self._init_model(ds, feature_stats, optimizer)

    def _init_model(self, ds, feature_stats: dict = None, optimizer=None):
        """ Initialize a model to predict multiple quantiles/percentiles that
        prevents the quantile crossing
        """

        n_quantiles = len(self.quantiles)

        all_inputs, encoded_features = layer_generator.model_input_layer(
            ds,
            features=self.config.features,
            embedding_dim=self.config.embedding_dim,
            feature_stats=feature_stats)

        x = tf.keras.layers.concatenate(encoded_features)

        for i, layer_size in enumerate(self.config.layer_sizes):
            x = layer_generator.composite_layer(
                x,
                layer_sizes=layer_size,
                l2=self.config.l2,
                dropout=self.config.dropout,
                dropout_min_layer_size=self.config.dropout_min_layer_size,
                batch_normalization=self.config.batch_normalization,
                name=f'cl_{i}')

        x1 = tfkl.Dense(1)(x)
        x2 = tfkl.Dense(n_quantiles - 1, activation='softplus')(x)
        x = tfkl.Concatenate(axis=-1)([x1, x2])

        # monotonically increasing outputs
        output = tfkl.Lambda(lambda y: tf.cumsum(y, axis=-1))(x)

        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=0.001)

        self.model = tf.keras.Model(all_inputs, output)
        self.model.compile(
            optimizer=optimizer,
            loss=pinball_loss(quantiles=self.quantiles))

    def evaluate_model(self, ds, log_dir: str, log_data: dict = None):
        """ Evaluate model """

        evaluate_percentile_model(
            model=self.model,
            ds=ds,
            log_dir=log_dir,
            log_data=log_data,
            quantiles=self.quantiles,
            qtile_range=self.quantile_range)
