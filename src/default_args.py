from model_config import ModelConfig, FeatureGroups

# Kept as a plain dict (`.to_dict()`) so it stays json-serialisable for the
# CLI overrides and logging; the model wrappers rebuild a ModelConfig from it.
model_args = ModelConfig(
    features=FeatureGroups(
        num_feats=['trip_distance', 'time',
                   'pickup_lon', 'pickup_lat', 'pickup_area',
                   'dropoff_lon', 'dropoff_lat', 'dropoff_area'],
        cat_int_feats=['passenger_count', 'vendor_id', 'weekday', 'month']),
    embedding_dim=10,
    layer_sizes=(32, (32, 32), 8),
    l2=.001,
    dropout=0,
    dropout_min_layer_size=12,
    batch_normalization=True).to_dict()

ds_args = dict(
    batch_size=2 ** 16,  # 65_536
    prefetch_size=-1,  # tf.data.AUTOTUNE
    cache=True,
    shuffle=True,  # permute the batch order at every epoch
    max_files=None,  # max number of files to use when loading a ds from dir
    take_size=-1)  # max number of rows in the dataset

callbacks_args = dict(
    histogram_freq=0,
    reduce_lr_patience=100,
    profile_batch=(10, 15),
    verbose=1,
    early_stopping_patience=250,
    period=10)

training_args = dict(
    epochs=100,
    verbose=0)
