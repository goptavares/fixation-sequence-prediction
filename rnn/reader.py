#!/usr/bin/python

"""
reader.py
Author: Gabriela Tavares, gtavares@caltech.edu

Adapted from tensorflow/models/tutorials/rnn/ptb/reader.py
[Copyright 2015 The TensorFlow Authors. All Rights Reserved.]
"""

import csv
import tensorflow as tf


def get_model_1_raw_data(train_file='model_1_train.csv',
                         valid_file='model_1_valid.csv',
                         test_file='model_1_test.csv'):
    train_data = []
    with open(train_file) as csvfile:
        reader = csv.DictReader(csvfile)
        seqCount = 0
        for row in reader:
            if row['data_point'] != seqCount:
                train_data.append(3)  # end of sequence
                seqCount = row['data_point']
            train_data.append(int(row['location']))

    valid_data = []
    with open(valid_file) as csvfile:
        reader = csv.DictReader(csvfile)
        seqCount = 0
        for row in reader:
            if row['data_point'] != seqCount:
                valid_data.append(3)  # end of sequence
                seqCount = row['data_point']
            valid_data.append(int(row['location']))

    test_data = []
    with open(test_file) as csvfile:
        reader = csv.DictReader(csvfile)
        seqCount = 0
        for row in reader:
            if row['data_point'] != seqCount:
                test_data.append(3)  # end of sequence
                seqCount = row['data_point']
            test_data.append(int(row['location']))

    return train_data, valid_data, test_data


def model_1_producer(raw_data, batch_size, num_steps, name=None):
    with tf.name_scope(name, "Model1Producer",
                       [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data",
                                        dtype=tf.int32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0 : batch_size * batch_len],
                          [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
          epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps],
                             [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                             [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])

        return x, y

