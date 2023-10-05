import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_data_validation
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.utils.class_weight import compute_class_weight
import more_itertools

import json
from box import ConfigBox
from ruamel.yaml import YAML

yaml = YAML(typ="safe")


def split_data():
    """
    Stratified train_val_test splits.
    """

    # Read DVC configuration
    params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))

    # Load TFDS dataset
    (bee_ds, ), info = tfds.load(
       name='bee_dataset',
        with_info=True,
        split=['train', ],
        as_supervised=True,
        shuffle_files=True,
    )

    # Drop all features except `varroa_output`
    bee_ds = bee_ds.map(
        lambda x, y: (x, tf.cast(y["varroa_output"], tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Calculate split sizes
    varroa_ds = bee_ds.filter(lambda x, y: tf.math.equal(y, 1.0))
    no_varroa_ds = bee_ds.filter(lambda x, y: tf.math.equal(y, 0.0))

    varroa_ds_size = len(list(varroa_ds.as_numpy_iterator()))
    no_varroa_ds_size = len(list(no_varroa_ds.as_numpy_iterator()))

    varroa_train_size = int(params.dataset.training.ratio * varroa_ds_size)
    no_varroa_train_size = int(params.dataset.training.ratio * no_varroa_ds_size)

    varroa_val_size = int(params.dataset.validation.ratio * varroa_ds_size)
    no_varroa_val_size = int(params.dataset.validation.ratio * no_varroa_ds_size)

    varroa_test_size = varroa_ds_size - (varroa_train_size + varroa_val_size)
    no_varroa_test_size = no_varroa_ds_size - (no_varroa_train_size + no_varroa_val_size)

    # Build splits
    train_ds = varroa_ds.take(varroa_train_size).concatenate(
        no_varroa_ds.take(no_varroa_train_size)
    )
    val_ds = varroa_ds.skip(varroa_train_size).take(varroa_val_size).concatenate(
        no_varroa_ds.skip(no_varroa_train_size).take(no_varroa_val_size)
    )
        
    test_ds = varroa_ds.skip(varroa_train_size + varroa_val_size).take(varroa_test_size).concatenate(
        no_varroa_ds.skip(no_varroa_train_size + no_varroa_val_size).take(no_varroa_test_size)
    )

    # Save splits
    train_ds.save(params.dataset.training.path, compression=None, shard_func=None, checkpoint_args=None)
    val_ds.save(params.dataset.validation.path, compression=None, shard_func=None, checkpoint_args=None)
    test_ds.save(params.dataset.testing.path, compression=None, shard_func=None, checkpoint_args=None)

if __name__ == "__main__":
    split_data()