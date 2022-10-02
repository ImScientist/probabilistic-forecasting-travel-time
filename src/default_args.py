model_args = dict(
    # Default model arguments applied to data with raw pickup and dropoff locations
    DSGRawLocation=dict(
        num_feats=['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                   'dropoff_latitude', 'trip_distance', 'time'],
        cat_int_feats=['weekday', 'month', 'pickup_area', 'dropoff_area', 'passenger_count'],
        cat_str_feats=['vendor_id'],
        emb_int_feats=[],
        emb_str_feats=[],
        embedding_dim=10,
        layer_sizes=(32, (32, 32), 8),
        l2=.001,
        dropout=0,
        dropout_min_layer_size=12,
        batch_normalization=True),
    # Default model arguments applied to data with anonymized pickup and dropoff locations
    DSGMaskedLocation=dict(
        num_feats=['trip_distance', 'time'],
        cat_int_feats=['weekday', 'month', 'passenger_count', 'vendor_id'],
        cat_str_feats=[],
        emb_int_feats=['pickup_location_id', 'dropoff_location_id'],
        emb_str_feats=[],
        embedding_dim=10,
        layer_sizes=(32, (32, 32), 8),
        l2=.001,
        dropout=0,
        dropout_min_layer_size=12,
        batch_normalization=True)
)

ds_args = dict(
    shuffle_buffer_size=0,
    batch_size=2 ** 20,  # 1_048_576,
    prefetch_size=-1,  # tf.data.AUTOTUNE,
    cache=True)

callbacks_args = dict(
    histogram_freq=0,
    reduce_lr_patience=100,
    profile_batch=(10, 15),
    verbose=0,
    early_stopping_patience=250,
    period=10)

training_args = dict(
    epochs=100,
    verbose=0)
