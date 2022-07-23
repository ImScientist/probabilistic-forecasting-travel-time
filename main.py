import io
import os

import logging
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model

from src import settings
from src.model import (
    df_to_dataset,
    get_model,
    train)

tfkl = tf.keras.layers
tfd = tfp.distributions

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def run(run_id: int):
    """ Train and save the model """

    log_dir = os.path.join(settings.TFBOARD_DIR, f'run_{run_id}')
    model_dir = os.path.join(settings.ARTIFACTS_DIR, f'run_{run_id}', 'model')
    weights_path = os.path.join(settings.ARTIFACTS_DIR, f'run_{run_id}', 'weights', 'weights')

    """
        Get data
    """
    path = os.path.join(settings.DATA_DIR, 'data_2016_5.parquet')
    df = pd.read_parquet(path).sample(300_000)

    idx_tr, idx_val_te = train_test_split(df.index, train_size=.8)
    idx_val, idx_te = train_test_split(idx_val_te, test_size=.5)
    del idx_val_te

    cluster = KMeans(n_clusters=settings.CLUSTERS)

    """
        Create datasets
    """
    ds_tr = df_to_dataset(df=df.loc[idx_tr],
                          cluster=cluster,
                          batch_size=settings.BATCH_SIZE)

    ds_val = df_to_dataset(df=df.loc[idx_val],
                           cluster=cluster,
                           batch_size=settings.BATCH_SIZE)

    ds_te = df_to_dataset(df=df.loc[idx_te],
                          cluster=cluster,
                          batch_size=settings.BATCH_SIZE)

    model = get_model(ds=ds_tr, dropout=settings.DROPOUT, l2=settings.L2)

    train(model=model,
          log_dir=log_dir,
          weights_path=weights_path,
          model_dir=model_dir,
          ds_tr=ds_tr,
          ds_val=ds_val,
          ds_te=ds_te)
