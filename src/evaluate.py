from __future__ import annotations

import io
import os
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

tfkl = tf.keras.layers
tfkc = tf.keras.callbacks
tfd = tfp.distributions
tfb = tfp.bijectors

dtype = tf.float32

logger = logging.getLogger(__name__)


def pretty_json(hp):
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))


def lognormal_pdf(loc, scale):
    """ Lognormal definition as in Wikipedia (FU scipy) """

    def pdf_fn(x):
        norm = np.sqrt(2 * np.pi) * scale * x
        arg = - (np.log(x) - loc) ** 2 / (2 * scale ** 2)
        return np.exp(arg) / norm

    return pdf_fn


def plot_to_image(figure):
    """
        Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this
        call.
    """

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    figure.savefig(buf, format='png')

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Add the batch dimension
    image = tf.expand_dims(image, 0)

    return image


def fig_mean_to_std_distribution(df: pd.DataFrame):
    """
        Plot the distribution of the mean to standard deviation ratios
        of the predicted distributions
    """

    label = 'mean to std ratio of predicted distributions'

    fig = plt.figure()
    plt.hist(df['mean'] / df['std'], bins=100, label=label)
    plt.legend()

    return fig


def fig_median_to_pct_range_distribution(
        df: pd.DataFrame, qtile_range: tuple[float, float]
):
    """
        Plot the distribution of the ratio btw the median and some percentile
        range.

    Params
    ------
      df: table with the columns:
          `q_050`: median
          `q_<quantile 1>`: first element of qtile_range
          `q_<quantile 2>`: second element of qtile_range
      qtile_range:
    """

    q1, q2 = qtile_range
    c1, c2 = f'q_{int(q1 * 100):03d}', f'q_{int(q2 * 100):03d}'
    label = f'median to ({c2} - {c1}) ratio of predicted distributions'

    fig = plt.figure()
    plt.hist(df['q_050'] / (df[c2] - df[c1]), bins=100, label=label)
    plt.legend()

    return fig


def fig_pct_skew(df: pd.DataFrame):
    """
        Plot distribution percentile of true values vs fraction of
        observations that belong to a lower percentile

    Params
    ------
      df: table with the columns:
          `pct`: percentile to which the observed value corresponds to
          `frac`: fraction of observations that belong to a lower predicted
              percentile
    """

    label = ('Predicted percentile vs fraction of observations '
             'that belong to a lower predicted percentile')

    fig = plt.figure(figsize=(15, 10))
    plt.scatter(df['pct'], df['frac'], alpha=0.2, s=20, label=label)
    plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')
    plt.legend()

    return fig


def fig_pct_skew_discrete(df: pd.DataFrame, quantiles: list):
    """
        For a discrete number of predicted percentiles calculate and visualize
        the fraction of observations that are below this percentile

    Params
    ------
      df: table with the columns:
         q_< pct >: predicted distribution percentiles
         y: observed value
    """

    frac = [(df['y'] < df[f'q_{int(q * 100):03d}']).mean()
            for q in quantiles]

    label = ('Predicted percentile vs fraction of observations \n'
             'that belong to a lower predicted percentile')

    fig = plt.figure()
    plt.scatter(quantiles, frac, label=label)
    plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')
    plt.legend()

    return fig


def evaluate_percentile_model(
        model,
        ds,
        log_dir: str,
        log_data: dict,
        quantiles: tuple,
        qtile_range: tuple[float, float]
):
    """ Compare predicted percentiles against observations """

    save_dir = os.path.join(log_dir, 'train')
    file_writer = tf.summary.create_file_writer(save_dir)

    df = pd.DataFrame()
    df['y'] = np.hstack(list(ds.map(lambda x, y: y).as_numpy_iterator()))

    pct_columns = [f'q_{int(x * 100):03d}' for x in quantiles]
    df[pct_columns] = model.predict(ds.map(lambda x, y: x))

    fig = fig_pct_skew_discrete(df, quantiles=quantiles)

    with file_writer.as_default():
        name = 'predicted pct vs frac of observations with lower predicted pct'
        img = plot_to_image(fig)
        tf.summary.image(name, img, step=0)

    fig = fig_median_to_pct_range_distribution(df, qtile_range=qtile_range)

    with file_writer.as_default():
        name = "median to pct-range of predicted distributions"
        img = plot_to_image(fig)
        tf.summary.image(name, img, step=0)

    with file_writer.as_default():
        log_data = pretty_json(log_data)
        tf.summary.text("experiment_args", log_data, step=0)


def evaluate_parametrized_pdf_model(
        model, ds, log_dir: str, log_data: dict, clusters: int = 20
):
    """ Compare predicted distributions against observations """

    save_dir = os.path.join(log_dir, 'train')
    file_writer = tf.summary.create_file_writer(save_dir)

    distribution_fn = model.layers[-1].function

    model_deterministic = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.layers[-2].output])

    df = pd.DataFrame()

    df['y'] = np.hstack(list(ds.map(lambda x, y: y).as_numpy_iterator()))

    # predict distribution parameters
    df[['p1', 'p2']] = model_deterministic.predict(ds.map(lambda x, y: x))

    distribution = distribution_fn(df[['p1', 'p2']].values)[0]

    df['mean'] = distribution.mean()
    df['std'] = distribution.stddev()
    df['pct'] = distribution.cdf(df[['y']].values)

    # fraction of prediction with lower percentile
    df = df.assign(frac=lambda x: (x['pct'].sort_values() * 0 + 1).cumsum() / x.shape[0])

    """
        Figures
    """

    fig = fig_mean_to_std_distribution(df=df)

    with file_writer.as_default():
        name = "mean to std ratio of predicted distributions"
        img = plot_to_image(fig)
        tf.summary.image(name, img, step=0)

    fig = fig_pct_skew(df.iloc[:1_000])

    with file_writer.as_default():
        name = 'predicted pct vs frac of observations with lower predicted pct'
        img = plot_to_image(fig)
        tf.summary.image(name, img, step=0)

    with file_writer.as_default():
        log_data = pretty_json(log_data)
        tf.summary.text("experiment_args", log_data, step=0)
