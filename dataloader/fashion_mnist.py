import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


class FashionMNISTInceptionDataset():
    def __init__(self, split=['train[:90%]', 'train[90%:]', 'test'], batch_size=128, buffer_size=2048):
        ds_train, ds_val, ds_test = tfds.load('fashion_mnist:3.0.1', split=split, shuffle_files=True)
        self.ds_train = self._build_train_pipeline(ds_train, batch_size, buffer_size)
        self.ds_val = self._build_test_pipeline(ds_val, batch_size)
        self.ds_test = self._build_test_pipeline(ds_test, batch_size)
        self.images = self._images(ds_test)

    def _build_train_pipeline(self, ds, batch_size, buffer_size):
        ds = ds.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def _build_test_pipeline(self, ds, batch_size):
        ds = ds.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def preprocess(self, element):
        images = (tf.cast(element['image'], tf.float32) - 127.5) / 127.5
        images = tf.image.grayscale_to_rgb(images)
        images = tf.image.resize(images, (75, 75))
        num_classes = 10
        one_hot_labels = tf.one_hot(element['label'], num_classes)
        return images, one_hot_labels

    def _images(self, ds):
        return np.concatenate([example['image'] for example in ds], axis=0)
