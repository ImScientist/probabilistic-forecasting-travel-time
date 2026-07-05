model_args = dict(
    num_feats=['trip_distance', 'time',
               'pickup_lon', 'pickup_lat', 'pickup_area',
               'dropoff_lon', 'dropoff_lat', 'dropoff_area'],
    cat_int_feats=['passenger_count', 'vendor_id', 'weekday', 'month'],
    cat_str_feats=[],
    emb_int_feats=[],
    emb_str_feats=[],
    embedding_dim=10,
    layer_sizes=(32, (32, 32), 8),
    l2=.001,
    dropout=0,
    dropout_min_layer_size=12,
    batch_normalization=True,
    learning_rate=1e-3,
    clipnorm=1.0,
    beta_2=0.95,
    lr_decay_steps=1000,
    lr_alpha=0.05)

ds_args = dict(
    batch_size=2 ** 20,  # 1_048_576
    prefetch_size=-1,  # tf.data.AUTOTUNE
    cache=True,
    shuffle=True,
    max_files=None,  # max number of files to use when loading a ds from dir
    take_size=-1)  # max number of elements in the dataset

callbacks_args = dict(
    histogram_freq=0,
    reduce_lr_patience=None,  # the cosine LR schedule in build_optimizer()
    # replaces this: combining ReduceLROnPlateau with a schedule rescales
    # (rather than overrides) the schedule's output, compounding into a
    # runaway LR decay
    profile_batch=(10, 15),
    verbose=1,
    early_stopping_patience=250,
    period=10)

training_args = dict(
    epochs=100,
    verbose=0)
