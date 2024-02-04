from __future__ import annotations

import os
import re
import logging
import tempfile
import numpy as np
import tensorflow as tf
from PIL import Image

import settings
from data.dataset import pq_to_dataset
from model.models import ModelWrapper

tfkc = tf.keras.callbacks

logger = logging.getLogger(__name__)


def remove_suffix(input_string, suffix):
    # .removesuffix() not supported in 3.8

    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


def log_model_architecture(model_, log_dir: str):
    """ Store a non-interactive readable model architecture """

    with tempfile.NamedTemporaryFile('w', suffix=".png") as temp:
        _ = tf.keras.utils.plot_model(
            model_,
            to_file=temp.name,
            show_shapes=True,
            dpi=64)

        im_frame = Image.open(temp.name)
        im_frame = np.asarray(im_frame)

        """ Log the figure """
        save_dir = os.path.join(log_dir, 'train')
        file_writer = tf.summary.create_file_writer(save_dir)

        with file_writer.as_default():
            tf.summary.image(
                "model summary",
                tf.constant(im_frame, dtype=tf.uint8)[tf.newaxis, ...],
                step=0)


def create_callbacks(
        log_dir: str,
        save_dir: str = None,
        histogram_freq: int = 0,
        reduce_lr_patience: int = 100,
        profile_batch: tuple = (10, 15),
        verbose: int = 0,
        early_stopping_patience: int = 250,
        period: int = 10
):
    """ Generate model training callbacks """

    callbacks = [
        tfkc.TensorBoard(
            log_dir=log_dir,
            histogram_freq=histogram_freq,
            profile_batch=profile_batch)]

    if reduce_lr_patience is not None:
        callbacks.append(
            tfkc.ReduceLROnPlateau(
                factor=0.2,
                patience=reduce_lr_patience,
                verbose=verbose))

    if early_stopping_patience is not None:
        callbacks.append(
            tfkc.EarlyStopping(patience=early_stopping_patience))

    if save_dir:
        path = os.path.join(
            save_dir,
            'checkpoints',
            'epoch_{epoch:03d}_loss_{val_loss:.4f}_cp.ckpt')

        callbacks.append(
            tfkc.ModelCheckpoint(
                path,
                save_weights_only=True,
                save_best_only=False,
                period=period))

    return callbacks


def get_best_checkpoint(
        checkpoint_dir: str,
        pattern=r'.*_loss_(\d+\.\d{4})_cp.ckpt.index'
):
    """ Parse names of all checkpoints, extract the validation loss
    and return the checkpoint with the lowest loss
    """

    pattern = r'.*_loss_(\d+\.\d{4})_cp.ckpt.index'

    checkpoints = os.listdir(checkpoint_dir)
    checkpoints = map(lambda x: re.fullmatch(pattern, x), checkpoints)
    checkpoints = filter(lambda x: x is not None, checkpoints)
    best_checkpoint = min(checkpoints, key=lambda x: float(x.group(1)))

    checkpoint_name = remove_suffix(best_checkpoint.group(0), suffix='.index')

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    return checkpoint_path


def get_experiment_id(logs_dir: str):
    """ Generate an unused experiment id by looking at the tensorboard entries """

    experiments = os.listdir(logs_dir)
    experiments = map(lambda x: re.fullmatch(r'ex_(\d{3})', x), experiments)
    experiments = filter(lambda x: x is not None, experiments)
    experiments = map(lambda x: int(x.group(1)), experiments)
    experiments = set(experiments)

    experiment_id = min(set(np.arange(1_000)) - experiments)

    logger.info(f'\n\nExperiment id: {experiment_id}\n\n')

    return experiment_id


def gpu_memory_setup():
    """ Restrict the amount of GPU memory that can be allocated by TensorFlow"""

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=settings.GPU_MEMORY_LIMIT * 1024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


def train(
        model_wrapper: type[ModelWrapper],
        ds_args: dict,
        model_args: dict,
        callbacks_args: dict,
        training_args: dict,
        data_preprocessed_dir: str
):
    """ Train and evaluate a model """

    os.makedirs(settings.TFBOARD_DIR, exist_ok=True)
    os.makedirs(settings.ARTIFACTS_DIR, exist_ok=True)

    gpu_memory_setup()

    experiment_id = get_experiment_id(settings.TFBOARD_DIR)

    log_dir = os.path.join(settings.TFBOARD_DIR, f'ex_{experiment_id:03d}')
    save_dir = os.path.join(settings.ARTIFACTS_DIR, f'ex_{experiment_id:03d}')

    # Initialize the training, validation and test datasets
    tr_dir = os.path.join(data_preprocessed_dir, 'train')
    va_dir = os.path.join(data_preprocessed_dir, 'validation')
    te_dir = os.path.join(data_preprocessed_dir, 'test')

    ds_tr = pq_to_dataset(data_dir=tr_dir, **ds_args)
    ds_va = pq_to_dataset(data_dir=va_dir, **ds_args)
    ds_te = pq_to_dataset(data_dir=te_dir, **ds_args)
    # ds_adapt = pq_to_dataset(data_dir=te_dir, cache=False, cycle_length=12)

    # Initialize the model generator
    mdl = model_wrapper(ds=ds_tr, **model_args)

    # Train and evaluate a model
    callbacks = create_callbacks(log_dir, save_dir, **callbacks_args)

    log_model_architecture(mdl.model, log_dir)

    mdl.model.fit(
        ds_tr, validation_data=ds_va, callbacks=callbacks, **training_args)

    # Save the model with the best weights
    checkpoint_path = get_best_checkpoint(os.path.join(save_dir, 'checkpoints'))
    mdl.model.load_weights(checkpoint_path)
    mdl.save(save_dir)

    mdl.model.evaluate(ds_tr)
    mdl.model.evaluate(ds_va)
    mdl.model.evaluate(ds_te)

    all_args = dict(
        model_wrapper=model_wrapper.__name__,
        model_args=model_args,
        dataset_args=ds_args,
        callbacks_args=callbacks_args,
        training_args=training_args)

    mdl.evaluate_model(ds=ds_te, log_dir=log_dir, log_data=all_args)
