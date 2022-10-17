# Probabilistic prediction of traveling times

We will use a neural network to predict the travelling time distribution between two locations. To predict a
distribution instead of a single value we modify the neural network by:

- feeding its outputs to the parameters of a probability distribution function (LogNormal oder Normal). Optimizing the
  model against the observed data is equivalent to minimizing the corresponding negative log-likelihood of the joint-pdf
  predicted by the model.

  <br><br>![Architecture](figs/nn_normal.png)<br><br>

- adding an extra layer with monotonically increasing outputs that represent a fixed set of the distribution-quantiles.
  Optimizing the model against the observed data is equivalent to minimizing the average of the pinball losses for every
  quantile.

  <br><br>![Architecture](figs/nn_iqf.png)<br><br>

We apply both models to the NYC taxi trip data that can be found
in the [nyc.gov](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) website.

## Collect & preprocess data

- We can collect the NYC taxi trip data (drives and taxi zones) for the entire 2016 from [nyc.gov](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) website, and store it in `DATA_DIR`:
  ```shell
  PYTHONPATH=$(pwd) python src/main.py collect-data --year=2016
  ```

- To generate features from the data and split it into a training, validation and test datasets execute:
  ```shell
  PYTHONPATH=$(pwd) python src/main.py preprocess-data \
    --tr=0.8 --va=0.1 --te=0.1
  ```
  It splits the data from every month into a training, validation and test dataset and stores it in a separate folder, as shown below:

  ```shell
  $DATA_DIR
  ├── raw/             # obtained in the data collection step
  └── preprocessed/    # obtained in the data preprocessing step
      ├── train/
      ├── validation/
      └── test/
  ```
  If you are worried about data drift over time you might need a different type of data splitting.


## Train model

- Change the following variables in `src/settings.py`:
  - `DATA_DIR`: raw and preprocessed data location   
  - `TFBOARD_DIR`: location of the logs visualized in tensorboard
  - `ARTIFACTS_DIR`: location where the model and dataset wrappers will be stored


- Train the model. The json strings that you can provide overwrite the default arguments used by the model:
  ```shell
  PYTHONPATH=$(pwd) python src/main.py train \
    --model_wrapper=ModelPDF \
    --model_args='{"l2": 0.0001, "batch_normalization": false, "layer_sizes": [64, [64, 64], [64, 64], 32, 8]}' \
    --ds_args='{"max_files": 2}' \
    --callbacks_args='{"period": 10, "profile_batch": 0}' \
    --training_args='{"epochs": 40}'
  ```

- After training the model the following directory structure will be generated (the experiment id is generated automatically):
  ```shell
  $ARTIFACTS_DIR
  └── ex_<id>
      ├── checkpoints/
      ├── model/
      └── model_attributes.json

  $TFBOARD_DIR
  └── ex_<id>
      ├── train/
      └── validation/
  ```

## Evaluate model

- We can evaluate accuracy and the uncertainty estimation provided by the predicted probability distribution functions or quantiles by using the following metrics/plots:

  - Fraction of cases (y-axis) where the observed percentile is below a given value (x-axis). Under observed percentile we understand the percentile of the predicted distribution to which the observation belongs to. For example, if we have predicted a normally distributed pdf with zero mean and unit standard deviation, and the observed value is 0, then the observed percentile will be 50. In the ideal case the plot should follow a straight line from (x,y) = (0,0) to (100,1).
  <br><br>

  - Histograms of the ratios between the mean and the standard deviation of the predicted distribution for every datapoint. For the model that predicts a fixed number of percentiles we replace the standard deviation with the difference between two percentiles, for example the 15-th and 85-th. This makes it harder to compare the predictions of both models.
     <br><br>
     <img src="figs/pdf-model_pct_plot.png" alt="isolated" height="250"/>
     <img src="figs/pdf-model_mean-to-std_histogram.png" alt="isolated" height="250"/>