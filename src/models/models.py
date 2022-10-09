import os
import json
import logging

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp

from abc import abstractmethod
from src.models import layer_generator
from src.evaluation import (
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
    tau = tf.constant(list(quantiles), dtype=dtype)

    def loss_fn(y, y_pred):
        return tfa.losses.pinball_loss(y, y_pred, tau=tau)

    return loss_fn


class ModelWrapper:
    def __init__(
            self,
            num_feats: list[str],
            cat_int_feats: list[str],
            cat_str_feats: list[str],
            emb_int_feats: list[str],
            emb_str_feats: list[str],
            embedding_dim: int = 10,
            layer_sizes: tuple | list = None,
            l2: float = .001,
            dropout: float = 0,
            dropout_min_layer_size: int = 12,
            batch_normalization: bool = False,
            **kwargs
    ):
        # input feats
        self.num_feats = num_feats
        self.cat_int_feats = cat_int_feats
        self.cat_str_feats = cat_str_feats
        self.emb_int_feats = emb_int_feats
        self.emb_str_feats = emb_str_feats
        self.embedding_dim = embedding_dim

        # nn parameters
        self.layer_sizes = layer_sizes or (32, 32, 8)
        self.l2 = l2
        self.dropout = dropout
        self.dropout_min_layer_size = dropout_min_layer_size
        self.batch_normalization = batch_normalization

        self.model = None
        self.custom_objects = None

    @abstractmethod
    def _init_model(self, ds):
        """ Initialize a model """

    def _load(self, load_dir: str):
        """ Load the model and all relevant class attributes """

        model_dir = os.path.join(load_dir, 'model')
        attributes_path = os.path.join(load_dir, 'model_attributes.json')

        self.model = tf.keras.models.load_model(
            model_dir,
            custom_objects=self.custom_objects)

        # Load other attributes
        with open(attributes_path, 'r') as f:
            attributes = json.load(f)

        for key, value in attributes.items():
            setattr(self, key, value)

    def save(self, save_dir: str):
        """ Save the model and other class attributes """

        model_dir = os.path.join(save_dir, 'model')
        attributes_path = os.path.join(save_dir, 'model_attributes.json')

        attributes = {k: getattr(self, k) for k in vars(self).keys()
                      if k not in ('model', 'custom_objects',)}

        self.model.save(model_dir, save_traces=False)

        with open(attributes_path, 'w') as f:
            json.dump(attributes, f, indent='\t')

    @abstractmethod
    def evaluate_model(self, ds, log_dir: str, log_data: dict = None):
        """ Evaluate model """


class ModelPDF(ModelWrapper):
    def __init__(
            self,
            num_feats: list[str],
            cat_int_feats: list[str],
            cat_str_feats: list[str],
            emb_int_feats: list[str],
            emb_str_feats: list[str],
            embedding_dim: int = 10,
            layer_sizes: tuple | list = None,
            l2: float = .001,
            dropout: float = 0,
            dropout_min_layer_size: int = 12,
            batch_normalization: bool = False,
            distribution: str = 'lognormal',
            ds=None,
            load_dir: str = None,
            **kwargs
    ):

        super(ModelPDF, self).__init__(
            num_feats=num_feats,
            cat_int_feats=cat_int_feats,
            cat_str_feats=cat_str_feats,
            emb_int_feats=emb_int_feats,
            emb_str_feats=emb_str_feats,
            embedding_dim=embedding_dim,
            layer_sizes=layer_sizes,
            l2=l2,
            dropout=dropout,
            dropout_min_layer_size=dropout_min_layer_size,
            batch_normalization=batch_normalization)

        assert ds is not None or load_dir is not None
        assert distribution in ('normal', 'lognormal')

        self.distribution = distribution
        self.custom_objects = {
            'neg_log_likelihood': neg_log_likelihood}

        if load_dir is not None:
            self._load(load_dir)
        else:
            self._init_model(ds)

    def _init_model(self, ds):
        """ Initialize a model where the second to last layer is used to
        parametrize a Normal/LogNormal distribution
        """

        all_inputs, encoded_features = layer_generator.model_input_layer(
            ds=ds,
            num_feats=self.num_feats,
            cat_int_feats=self.cat_int_feats,
            cat_str_feats=self.cat_str_feats,
            emb_int_feats=self.emb_int_feats,
            emb_str_feats=self.emb_str_feats,
            embedding_dim=self.embedding_dim)

        x = tf.keras.layers.concatenate(encoded_features)

        for i, layer_size in enumerate(self.layer_sizes):
            x = layer_generator.composite_layer(
                x,
                layer_sizes=layer_size,
                l2=self.l2,
                dropout=self.dropout,
                dropout_min_layer_size=self.dropout_min_layer_size,
                batch_normalization=self.batch_normalization,
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

        self.model = tf.keras.Model(all_inputs, output)
        self.model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.01),
            loss=neg_log_likelihood)

    def evaluate_model(self, ds, log_dir: str, log_data: dict = None):
        """ Evaluate model """

        evaluate_parametrized_pdf_model(
            self.model, ds=ds, log_dir=log_dir, log_data=log_data)


class ModelIQF(ModelWrapper):
    def __init__(
            self,
            num_feats: list[str],
            cat_int_feats: list[str],
            cat_str_feats: list[str],
            emb_int_feats: list[str],
            emb_str_feats: list[str],
            embedding_dim: int = 10,
            layer_sizes: tuple | list = None,
            l2: float = .001,
            dropout: float = 0,
            dropout_min_layer_size: int = 12,
            batch_normalization: bool = False,
            quantiles: tuple = (.05, .15, .3, .5, .7, .85, .95),
            quantile_range: tuple = (.15, .85),
            ds=None,
            load_dir: str = None,
            **kwargs
    ):
        super(ModelIQF, self).__init__(
            num_feats=num_feats,
            cat_int_feats=cat_int_feats,
            cat_str_feats=cat_str_feats,
            emb_int_feats=emb_int_feats,
            emb_str_feats=emb_str_feats,
            embedding_dim=embedding_dim,
            layer_sizes=layer_sizes,
            l2=l2,
            dropout=dropout,
            dropout_min_layer_size=dropout_min_layer_size,
            batch_normalization=batch_normalization)

        assert ds is not None or load_dir is not None

        self.quantiles = quantiles
        self.quantile_range = quantile_range
        self.custom_objects = {
            'loss_fn': pinball_loss(quantiles=self.quantiles)}

        if load_dir is not None:
            self._load(load_dir)
        else:
            self._init_model(ds)

    def _init_model(self, ds):
        """ Initialize a model to predict multiple quantiles/percentiles that
        prevents the quantile crossing
        """

        n_quantiles = len(self.quantiles)

        all_inputs, encoded_features = layer_generator.model_input_layer(
            ds,
            num_feats=self.num_feats,
            cat_int_feats=self.cat_int_feats,
            cat_str_feats=self.cat_str_feats,
            emb_int_feats=self.emb_int_feats,
            emb_str_feats=self.emb_str_feats,
            embedding_dim=self.embedding_dim)

        x = tf.keras.layers.concatenate(encoded_features)

        for i, layer_size in enumerate(self.layer_sizes):
            x = layer_generator.composite_layer(
                x,
                layer_sizes=layer_size,
                l2=self.l2,
                dropout=self.dropout,
                dropout_min_layer_size=self.dropout_min_layer_size,
                batch_normalization=self.batch_normalization,
                name=f'cl_{i}')

        x1 = tfkl.Dense(1)(x)
        x2 = tfkl.Dense(n_quantiles - 1, activation='softplus')(x)
        x = tfkl.Concatenate(axis=-1)([x1, x2])

        # monotonically increasing outputs
        output = tfkl.Lambda(lambda y: tf.cumsum(y, axis=-1))(x)

        self.model = tf.keras.Model(all_inputs, output)
        self.model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.01),
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
