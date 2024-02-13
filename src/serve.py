import os
import logging
import tensorflow as tf

from model.models import ModelPDF

logger = logging.getLogger(__name__)


def prepare_servable_mean_std(load_dir: str):
    """ Modify the model that outputs samples from a parametrized distribution
    to a model that outputs the mean and std of the distribution """

    servable_dir = os.path.join(load_dir, 'model_mean_std')

    mdl = ModelPDF(load_dir=load_dir)

    input_signature = mdl.model.signatures['serving_default'].structured_input_signature[1]

    @tf.function
    def mean_value(**kwargs):
        return {'mean_value': mdl.model(kwargs).mean()}

    @tf.function
    def std(**kwargs):
        return {'std': mdl.model(kwargs).stddev()}

    fn_mean = mean_value.get_concrete_function(**input_signature)
    fn_std = std.get_concrete_function(**input_signature)

    logger.info(f'Store servable in {servable_dir}')

    tf.saved_model.save(
        mdl.model,
        servable_dir,
        signatures={
            'mean_value': fn_mean,
            'std': fn_std},
        options=None)
