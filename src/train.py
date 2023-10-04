import os
import shutil
from pathlib import Path
import json
from box import ConfigBox
from ruamel.yaml import YAML

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

tfkl = tf.keras.layers
tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors

from dvclive import Live
from dvclive.keras import DVCLiveCallback

yaml = YAML(typ="safe")


def train():
    # Read DVC configuration
    params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))

    # tf.random.set_seed(params.base.random_seed)

    # Configure MultiWorkerMirroredStrategy with TensorFlow's ring algorithms for all-reduce and all-gather.
    implementation = tf.distribute.experimental.CommunicationImplementation.RING
    communication_options = tf.distribute.experimental.CommunicationOptions(implementation=implementation)
    strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)

    # Get current worker's task_type and task_id
    task_type, task_id = (
        strategy.cluster_resolver.task_type,
        strategy.cluster_resolver.task_id,
    )

    # Configure distributed training and test pipelines
    global_batch_size = params.train.batch_size_per_replica * strategy.num_replicas_in_sync

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    (train_ds, test_ds), info = tfds.load(
        name='bee_dataset',
        with_info=True,
        # data_dir='/tensorflow_datasets',
        split=[
            'train[:{}%]'.format(params.train.training_split), 
            'train[:{}%]'.format(params.train.training_split)
        ],
        as_supervised=True,
        shuffle_files=True,
    )
    
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(tf.data.AUTOTUNE)   # train_ds.cardinality()
    train_ds = train_ds.repeat(params.train.epochs) 
    train_ds = train_ds.map(lambda x, y: (x, y["varroa_output"]), num_parallel_calls=tf.data.AUTOTUNE) 
    train_ds = train_ds.batch(global_batch_size) 
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    train_ds = train_ds.with_options(options)

    test_ds = test_ds.cache() 
    test_ds = test_ds.shuffle(tf.data.AUTOTUNE) 
    test_ds = test_ds.repeat(params.train.epochs) 
    test_ds = test_ds.map(lambda x, y: (x, y["varroa_output"]), num_parallel_calls=tf.data.AUTOTUNE) 
    test_ds = test_ds.batch(global_batch_size) 
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.with_options(options) 

    # see: https://github.com/rmrschub/varroa_classifier_notebooks/blob/master/dataset_exploration.ipynb
    class_weights={
        0: 0.60170308, 
        1: 2.95813586
    }

    # Define, build and compile model within strategy scope
    with strategy.scope():
        model = tf.keras.applications.EfficientNetV2S(
            include_top=True,
            weights=None,
            input_shape=info.features['input'].shape,
            pooling=params.model.pooling,
            classes=1,
            classifier_activation='sigmoid',
            include_preprocessing=True,
        )

        optimizer = tf.optimizers.Adam(
            learning_rate=params.train.learning_rate
        )

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    # Train distributed model using distributed training dataset
    if task_id==0:
        with Live() as live:
            model.fit(
                train_ds,
                # epochs=params.train.epochs, 
                # batch_size=global_batch_size,
                # validation_data=validation_ds,
                # validation_freq=params.train.validation_freq,
                # validation_steps=None,
                verbose=params.train.verbosity,
                callbacks=[DVCLiveCallback(live=live), ],
                class_weight=class_weights
            )
    else:
        model.fit(
            train_ds,
            # epochs=params.train.epochs, 
            # batch_size=global_batch_size,
            # validation_data=validation_ds,
            # validation_freq=params.train.validation_freq,
            # validation_steps=None,
            verbose=params.train.verbosity,
            class_weight=class_weights
        )

    if task_id==0:
        model.evaluate(
            test_ds,
            verbose=params.train.verbosity,
        )

    # Save trained model(s)
    model.save_weights(
        'model_{}.keras'.format(task_id)
    )

if __name__ == "__main__":
    train()