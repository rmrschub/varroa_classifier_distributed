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
    
    # Configure cross device communication 
    implementation = tf.distribute.experimental.CommunicationImplementation.RING
    communication_options = tf.distribute.experimental.CommunicationOptions(implementation=implementation)

    # Define distribution strategy for synchronous training on multiple workers.
    strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)
    global_batch_size = params.train.batch_size_per_replica * strategy.num_replicas_in_sync

    with strategy.scope():
        # Define, build and compile model within strategy scope
        model = tf.keras.applications.EfficientNetV2S(
            include_top=True,
            weights=None,
            input_shape=params.model.input_shape,
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

        # Configure distributed training and test pipelines
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        global_batch_size = params.train.batch_size_per_replica * strategy.num_replicas_in_sync
        train_ds = tf.data.Dataset.load(params.dataset.training.path)
        train_ds = train_ds.cache()
        train_ds = train_ds.shuffle(10 * global_batch_size)
        train_ds = train_ds.batch(global_batch_size) 
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        train_ds = train_ds.with_options(options)

        val_ds = tf.data.Dataset.load(params.dataset.validation.path)
        val_ds = val_ds.cache() 
        val_ds = val_ds.shuffle(10 * global_batch_size)   
        val_ds = val_ds.batch(global_batch_size) 
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.with_options(options) 

    # Train distributed model using distributed training dataset   
    class_weights= {
        # see: https://github.com/rmrschub/varroa_classifier_notebooks/blob/master/dataset_exploration.ipynb
        0: 0.60170308, 
        1: 2.95813586
    }

    # Get current worker's task_type and task_id
    task_type, task_id = (
        strategy.cluster_resolver.task_type,
        strategy.cluster_resolver.task_id,
    )

    if task_id==0:
        with Live() as live:
            model.fit(
                train_ds,
                epochs=params.train.epochs, 
                batch_size=global_batch_size,
                validation_data=val_ds,
                validation_freq=params.train.validation_freq,
                validation_steps=None,
                verbose=params.train.verbosity,
                callbacks=[DVCLiveCallback(live=live), ],
                class_weight=class_weights
            )
    else:
        # worker node
        model.fit(
            train_ds,
            epochs=params.train.epochs, 
            batch_size=global_batch_size,
            validation_data=val_ds,
            validation_freq=params.train.validation_freq,
            validation_steps=None,
            verbose=params.train.verbosity,
            class_weight=class_weights
        )

    # Save trained model(s)
    write_model_path = 'model_{}'.format(task_id) if task_id==0 else 'tmp/model_{}'.format(task_id)
    model.save(write_model_path)

if __name__ == "__main__":
    train()