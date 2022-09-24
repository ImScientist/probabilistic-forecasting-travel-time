import os
import json
import joblib
import logging
import tempfile
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from PIL import Image

from src.evaluation import (
    evaluate_percentile_model,
    evaluate_parametrized_pdf_model)

from src.model import (
    df_to_dataset,
    get_model,
    get_model_iqf)

tfkl = tf.keras.layers
tfkc = tf.keras.callbacks
tfd = tfp.distributions
tfb = tfp.bijectors

dtype = tf.float32

logger = logging.getLogger(__name__)


def train_val_test_split(df_index: pd.Index):
    """ Split data into training 80% validation 10% and test 10% parts """

    idx_tr, idx_val_te = train_test_split(df_index, train_size=.8)
    idx_val, idx_te = train_test_split(idx_val_te, test_size=.5)
    del idx_val_te

    return idx_tr, idx_val, idx_te


def store_model_architecture(model, log_dir: str):
    """ Store a non-interactive readable model architecture """

    with tempfile.NamedTemporaryFile('w', suffix=".png") as temp:
        _ = tf.keras.utils.plot_model(
            model,
            to_file=temp.name,
            show_shapes=True,
            dpi=64)

        im_frame = Image.open(temp.name)
        im_frame = np.asarray(im_frame)

        """ Log the figure """
        save_dir = os.path.join(log_dir, 'train')
        file_writer = tf.summary.create_file_writer(save_dir)

        with file_writer.as_default():
            tf.summary.image("model summary",
                             tf.constant(im_frame, dtype=tf.uint8)[tf.newaxis, ...],
                             step=0)


def train_evaluate(
        model,
        cluster: KMeans,
        ds_tr,
        ds_va,
        ds_te,
        log_dir: str,
        quantiles: list = None,
        qtile_range: tuple[int, int] = None,
        save_dir: str = None,
        epochs: int = 100,
        early_stopping_patience: int = 250,
        reduce_lr_patience: int = 100,
        histogram_freq: int = 0,
        profile_batch: tuple = (10, 15),
        verbose: int = 0,
        evaluate: bool = True
):
    """ Train and evaluate a model """

    callbacks = [
        tfkc.TensorBoard(log_dir=log_dir,
                         histogram_freq=histogram_freq,
                         profile_batch=profile_batch)
    ]

    if reduce_lr_patience is not None:
        callbacks.append(
            tfkc.ReduceLROnPlateau(factor=0.2,
                                   patience=reduce_lr_patience,
                                   verbose=verbose))

    if early_stopping_patience is not None:
        callbacks.append(
            tfkc.EarlyStopping(patience=early_stopping_patience))

    if save_dir:
        path = os.path.join(save_dir, 'checkpoints', 'cp.ckpt')
        callbacks.append(
            tfkc.ModelCheckpoint(path,
                                 save_weights_only=True,
                                 save_best_only=True))

    store_model_architecture(model, log_dir)

    _ = model.fit(
        ds_tr,
        validation_data=ds_va,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose)

    # Load the best model weights and save the full model
    if save_dir:
        cluster_path = os.path.join(save_dir, 'cluster.joblib')
        checkpoint_path = os.path.join(save_dir, 'checkpoints', 'cp.ckpt')
        quantiles_path = os.path.join(save_dir, 'quantiles.json')
        model_dir = os.path.join(save_dir, 'model')

        joblib.dump(cluster, cluster_path)
        model.load_weights(checkpoint_path)
        model.save(model_dir, save_traces=False)

        if quantiles is not None:
            with open(quantiles_path, 'w') as f:
                json.dump(quantiles, f, indent=4)

    model.evaluate(ds_tr)
    model.evaluate(ds_va)
    model.evaluate(ds_te)

    if evaluate and quantiles is None:
        evaluate_parametrized_pdf_model(model, ds_te.take(1), log_dir)
    elif evaluate and quantiles is not None:
        evaluate_percentile_model(model, ds_te.take(1), log_dir, quantiles, qtile_range)


def run(
        data_path: str,
        main_dir: str,
        ex: int,
        use_percentile_model: bool = False,
        epochs: int = 1_000
):
    """ lalala """

    clusters = 20
    batch_size = 2 ** 20  # 1_048_576
    prefetch_size = tf.data.AUTOTUNE

    log_dir = os.path.join(main_dir, 'tfboard', f'ex_{ex:02d}')
    save_dir = os.path.join(main_dir, 'saved_models', f'ex_{ex:02d}')

    cluster = KMeans(n_clusters=clusters)

    df = pd.read_parquet(data_path)

    idx_tr, idx_val, idx_te = train_val_test_split(df.index)

    ds_tr = df_to_dataset(
        df=df.loc[idx_tr],
        cluster=cluster,
        shuffle_buffer_size=0,  # noqa
        batch_size=batch_size,
        prefetch_size=prefetch_size,
        cache=True)

    ds_val = df_to_dataset(
        df=df.loc[idx_val],
        cluster=cluster,
        shuffle_buffer_size=0,  # noqa
        batch_size=batch_size,
        prefetch_size=prefetch_size,
        cache=True)

    ds_te = df_to_dataset(
        df=df.loc[idx_te],
        cluster=cluster,
        shuffle_buffer_size=0,  # noqa
        batch_size=batch_size,
        prefetch_size=prefetch_size,
        cache=True)

    if use_percentile_model:
        quantiles = (.1, .3, .5, .7, .9)
        qtile_range = (.3, .7)

        model = get_model_iqf(
            ds=ds_tr,
            layer_sizes=(32, (32, 32), 8),
            l2=0.001,  # noqa
            dropout=0,  # noqa
            dropout_min_layer_size=14,
            batch_normalization=True,
            quantiles=quantiles)
    else:
        quantiles = None
        qtile_range = None

        model = get_model(
            ds=ds_tr,
            layer_sizes=(32, (32, 32), 8),
            l2=0.001,  # noqa
            dropout=0,  # noqa
            dropout_min_layer_size=14,
            batch_normalization=True,
            distribution='lognormal')

    train_evaluate(
        model=model,
        cluster=cluster,
        ds_tr=ds_tr,
        ds_va=ds_val,
        ds_te=ds_te,
        log_dir=log_dir,
        quantiles=quantiles,
        qtile_range=qtile_range,
        save_dir=save_dir,
        epochs=epochs,
        early_stopping_patience=250,  # noqa
        reduce_lr_patience=100,  # noqa
        histogram_freq=0,  # noqa
        profile_batch=0,  # noqa
        verbose=1)


if __name__ == '__main__':
    """
    PYTHONPATH=$(pwd) python src/training.py
    """

    data_path = '/home/ai/Data/nyc_taxi/data_2016_5.parquet'
    main_dir = '/home/ai/projects/carrrrs_ny'
    ex = 9
    use_percentile_model = False

    run(
        data_path=data_path,
        main_dir=main_dir,
        ex=ex,
        use_percentile_model=use_percentile_model)
