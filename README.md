# Probabilistic prediction of traveling times

We will use a neural network to predict the travelling time distribution between two locations. To predict a
distribution instead of a single value we modify the neural network by:

- feeding its outputs to the parameters of a probability distribution function (LogNormal oder Normal). Optimizing the
  model against the observed data is equivalent to minimizing the corresponding negative log-likelihood of the joint-pdf
  predicted by the model.

  ![Architecture](figs/nn_normal.png)

- adding an extra layer with monotonically increasing outputs that represent a fixed set of the distribution-quantiles.
  Optimizing the model against the observed data is equivalent to minimizing the average of the pinball losses for every
  quantile.

  ![Architecture](figs/nn_iqf.png)

We apply both models to the NYC taxi trip data that can be found as
a [BigQuery public dataset](https://cloud.google.com/datasets) hosted by Google. Recently, the exact pickup/drop-off
locations were replaced location-ids. The model supports both options depending on what data do you have.

## Collect data from BigQuery

- Change all variables in `src/settings.py`:
    - `PROJECT_ID`: the project id in Google cloud that will be billed for the execution of the data collection queries
    - `DATA_DIR` is the location where the data will be stored


- Assuming that you already have the gcloud sdk execute the following commands:
  ```shell
  gcloud init
  gcloud auth login
  ```


- The following command will download the relevant columns from NYC taxi trip dataset for the entire 2016. You will be
  billed for less the 1TB of data which is within the monthly limits of
  the [Free usage tier](https://cloud.google.com/bigquery/pricing#free-tier):
  ```shell
  PYTHONPATH=$(pwd) python src/main.py collect-data
  ```

## Train model

- Change the following variables in `src/settings.py`:
    - `TFBOARD_DIR`: location of the logs visualized in tensorboard
    - `ARTIFACTS_DIR`: location where the model and dataset wrappers will be stored

- Train the model. The json strings that you can provide overwrite the default arguments used by the model:
  ```shell
  PYTHONPATH=$(pwd) python src/main.py train \
    --dataset_generator=DSGMaskedLocation \
    --model_wrapper=ModelPDF \
    --model_args='{"l2": 0.001, "batch_normalization": false, "embedding_dim": 16}' \
    --callbacks_args='{"period": 10, "profile_batch": 0}' \
    --training_args='{"epochs": 40}'
  ```
