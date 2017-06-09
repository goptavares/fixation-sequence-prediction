#!/usr/bin/python

"""
reader.py
Author: Gabriela Tavares, gtavares@caltech.edu

Adapted from tensorflow/tensorflow/contrib/learn/python/learn/datasets/mnist.py
[Copyright 2016 The TensorFlow Authors. All Rights Reserved.]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed

import pandas as pd


def dense_to_one_hot(input_dense, num_classes):
  """Convert input from scalars to one-hot vectors."""
  input_size = input_dense.shape[0]
  index_offset = numpy.arange(input_size) * num_classes
  input_one_hot = numpy.zeros((input_size, num_classes))
  input_one_hot.flat[index_offset + input_dense.ravel()] = 1
  return input_one_hot


def extract_sequences(data_path, num_classes=9):

  df = pd.DataFrame.from_csv(data_path, header=0, sep=",", index_col=None)

  data_points = df.data_point.unique()

  data = []
  for d in data_points:
    fixations = numpy.array(
        df.loc[df["data_point"]==d, ["location"]])
    # data.append(dense_to_one_hot(fixations[:160], num_classes))
    data.append(fixations[:160])

  data = numpy.array(data)
  data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)

  return data


class DataSet(object):

  def __init__(self,
               sequences,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid dtype %r, expected uint8 or float32' % dtype)

    self._num_examples = sequences.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    if reshape:
      assert sequences.shape[3] == 1
      sequences = sequences.reshape(sequences.shape[0],
                                    sequences.shape[1] * sequences.shape[2])
    # if dtype == dtypes.float32:
    #   # Convert from [0, 255] -> [0.0, 1.0].
    #   sequences = sequences.astype(numpy.float32)
    #   sequences = numpy.multiply(sequences, 1.0 / 255.0)

    self._sequences = sequences
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def sequences(self):
    return self._sequences

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._sequences = self.sequences[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._sequences[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._sequences = self.sequences[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._sequences[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._sequences[start:end]


def read_data_sets(train_data_path,
                   test_data_path,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=50,
                   seed=None):

  train_sequences = extract_sequences(train_data_path)
  test_sequences = extract_sequences(test_data_path)

  if not 0 <= validation_size <= len(train_sequences):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_sequences), validation_size))

  validation_sequences = train_sequences[:validation_size]
  train_sequences = train_sequences[validation_size:]

  options = dict(dtype=dtype, reshape=reshape, seed=seed)
  
  train = DataSet(train_sequences, **options)
  validation = DataSet(validation_sequences, **options)
  test = DataSet(test_sequences, **options)
  
  return base.Datasets(train=train, validation=validation, test=test)


def load_data(train_data_path, test_data_path):
  return read_data_sets(train_data_path, test_data_path)

