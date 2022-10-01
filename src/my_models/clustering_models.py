import os
import json
import joblib
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans

from src.dataset import df_to_dataset_01
from src.evaluation import evaluate_parametrized_pdf_model, evaluate_percentile_model
from src.models import losses
from src.models.models import get_model_parametrized_dist, get_model_iqf
from src.my_models.base import MyModel


class MyModelClusteredLocation(MyModel):
    """ Model that preprocesses the data by mapping the pickup- and dropoff
    coordinates to an area-id (areas specified through a k-means clustering
    model) """

    def __init__(
            self,
            area_clusters: int = 20,
            layer_sizes: tuple = None,
            l2: float = .001,
            dropout: float = 0,
            dropout_min_layer_size: int = 12,
            batch_normalization: bool = False
    ):
        super(MyModelClusteredLocation, self).__init__()

        # feature preprocessing params
        self.num_feats = [
            'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
            'dropoff_latitude', 'trip_distance', 'time']

        self.cat_int_feats = [
            'weekday', 'month', 'pickup_area',
            'dropoff_area', 'passenger_count']

        self.cat_str_feats = ['vendor_id']

        # nn parameters
        self.layer_sizes = layer_sizes or (32, 32, 8)
        self.l2 = l2
        self.dropout = dropout
        self.dropout_min_layer_size = dropout_min_layer_size
        self.batch_normalization = batch_normalization

        self.cluster = KMeans(n_clusters=area_clusters)
        self.model = None

    def df_to_dataset(
            self,
            df: pd.DataFrame,
            shuffle_buffer_size: int = 0,
            batch_size: int = None,
            prefetch_size: int = None,
            cache: bool = False,
            **kwargs
    ):
        """ Create a dataset from a dataframe """

        ds = df_to_dataset_01(
            df,
            cluster=self.cluster,
            shuffle_buffer_size=shuffle_buffer_size,
            batch_size=batch_size,
            prefetch_size=prefetch_size,
            cache=cache)

        return ds

    def save(self, save_dir: str):
        """ Save the model and other class attributes """

        model_dir = os.path.join(save_dir, 'model')
        cluster_path = os.path.join(save_dir, 'cluster.joblib')
        attributes_path = os.path.join(save_dir, 'attributes.json')

        attributes = {k: getattr(self, k) for k in vars(self).keys()
                      if k not in ('cluster', 'model')}

        self.model.save(model_dir, save_traces=False)
        joblib.dump(self.cluster, cluster_path)

        with open(attributes_path, 'w') as f:
            json.dump(attributes, f, indent='\t')


class MyModelClusteredLocationParametrizedPDF(MyModelClusteredLocation):
    """ Model that feeds a NN into a parametrized distribution (Normal or
    LogNormal); It also preprocesses the data by mapping the pickup- and
    dropoff coordinates to an area-id (areas specified through a k-means
    clustering model) """

    def __init__(
            self,
            area_clusters: int = 20,
            layer_sizes: tuple = None,
            l2: float = .001,
            dropout: float = 0,
            dropout_min_layer_size: int = 12,
            batch_normalization: bool = False,
            distribution: str = 'lognormal'
    ):
        assert distribution in ('lognormal', 'normal')

        self.distribution = distribution

        super(MyModelClusteredLocationParametrizedPDF, self).__init__(
            area_clusters=area_clusters,
            layer_sizes=layer_sizes,
            l2=l2,
            dropout=dropout,
            dropout_min_layer_size=dropout_min_layer_size,
            batch_normalization=batch_normalization)

    def init_model(self, ds, **kwargs):
        """ Initialize a model """

        self.model = get_model_parametrized_dist(
            ds=ds,
            num_feats=self.num_feats,
            cat_int_feats=self.cat_int_feats,
            cat_str_feats=self.cat_str_feats,
            layer_sizes=self.layer_sizes,
            l2=self.l2,
            dropout=self.dropout,
            dropout_min_layer_size=self.dropout_min_layer_size,
            batch_normalization=self.batch_normalization,
            distribution=self.distribution)

    def load(self, load_dir: str):
        """ Load the model and all relevant class attributes """

        model_dir = os.path.join(load_dir, 'model')
        cluster_path = os.path.join(load_dir, 'cluster.joblib')
        attributes_path = os.path.join(load_dir, 'attributes.json')

        self.cluster = joblib.load(cluster_path)
        self.model = tf.keras.models.load_model(
            model_dir,
            custom_objects={'neg_log_likelihood': losses.neg_log_likelihood})

        # Load other attributes
        with open(attributes_path, 'r') as f:
            attributes = json.load(f)

        for key, value in attributes.items():
            setattr(self, key, value)

    def evaluate_model(self, ds, log_dir: str):
        """ Evaluate model """

        evaluate_parametrized_pdf_model(
            model=self.model,
            ds=ds,
            log_dir=log_dir)


class MyModelClusteredLocationIQF(MyModelClusteredLocation):
    """ Model uses a NN to predict different quantiles of the target variable;
    It also preprocesses the data by mapping the pickup- and dropoff
    coordinates to an area-id (areas specified through a k-means clustering
    model)
    """

    def __init__(
            self,
            area_clusters: int = 20,
            layer_sizes: tuple = None,
            l2: float = .001,
            dropout: float = 0,
            dropout_min_layer_size: int = 12,
            batch_normalization: bool = False,
            quantiles: tuple = (.1, .3, .5, .7, .9),
            quantile_range: tuple = (.3, .7)
    ):
        self.quantiles = quantiles
        self.quantile_range = quantile_range

        super(MyModelClusteredLocationIQF, self).__init__(
            area_clusters=area_clusters,
            layer_sizes=layer_sizes,
            l2=l2,
            dropout=dropout,
            dropout_min_layer_size=dropout_min_layer_size,
            batch_normalization=batch_normalization)

    def init_model(self, ds, **kwargs):
        """ Initialize a model """

        self.model = get_model_iqf(
            ds=ds,
            num_feats=self.num_feats,
            cat_int_feats=self.cat_int_feats,
            cat_str_feats=self.cat_str_feats,
            layer_sizes=self.layer_sizes,
            l2=self.l2,
            dropout=self.dropout,
            dropout_min_layer_size=self.dropout_min_layer_size,
            batch_normalization=self.batch_normalization,
            quantiles=self.quantiles)

    def load(self, load_dir: str):
        """ Load the model and all relevant class attributes """

        model_dir = os.path.join(load_dir, 'model')
        cluster_path = os.path.join(load_dir, 'cluster.joblib')
        attributes_path = os.path.join(load_dir, 'attributes.json')

        # Load other attributes
        with open(attributes_path, 'r') as f:
            attributes = json.load(f)

        for key, value in attributes.items():
            setattr(self, key, value)

        self.cluster = joblib.load(cluster_path)

        loss_fn = losses.pinball_loss(quantiles=self.quantiles)

        self.model = tf.keras.models.load_model(
            model_dir,
            custom_objects={'loss_fn': loss_fn})

    def evaluate_model(self, ds, log_dir: str):
        """ Evaluate model """

        evaluate_percentile_model(
            model=self.model,
            ds=ds,
            log_dir=log_dir,
            quantiles=self.quantiles,
            qtile_range=self.quantile_range)
