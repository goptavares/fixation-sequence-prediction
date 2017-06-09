#!/usr/bin/python

"""
sample_lstm.py
Author: Gabriela Tavares, gtavares@caltech.edu

To run:
$ python sample_lstm.py --model_path=my_model-10908.meta --model2=test
"""

import inspect
import numpy as np
import tensorflow as tf
import time

from lstm import SmallConfig, MediumConfig, LargeConfig, TestConfig

flags = tf.flags

flags.DEFINE_string("model_path", None, "Model directory.")
flags.DEFINE_string(
    "model2", "small",
    "A type of model. Possible options are: small, medium, large.")

FLAGS = flags.FLAGS

def data_type():
  return tf.float32


def get_config():
    if FLAGS.model2 == "small":
        return SmallConfig()
    elif FLAGS.model2 == "medium":
        return MediumConfig()
    elif FLAGS.model2 == "large":
        return LargeConfig()
    elif FLAGS.model2 == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model2)


def main(_):
    config = get_config()

    hidden_size = config.hidden_size

    vocab_size = config.vocab_size
    num_layers = config.num_layers

    seed_size = 10
    max_seq_length = 50

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to
    # be different than reported in the paper.
    def lstm_cell():
        # With the latest TensorFlow source code (as of Mar 27, 2017),
        # the BasicLSTMCell will need a reuse parameter which is
        # unfortunately not defined in TensorFlow 1.0. To maintain
        # backwards compatibility, we add an argument check here:
        if 'reuse' in inspect.getargspec(
            tf.contrib.rnn.BasicLSTMCell.__init__).args:
            return tf.contrib.rnn.BasicLSTMCell(
                hidden_size, forget_bias=0.0, state_is_tuple=True, reuse=None)
        else:
            return tf.contrib.rnn.BasicLSTMCell(
                hidden_size, forget_bias=0.0, state_is_tuple=True)

    cell = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell() for _ in range(num_layers)], state_is_tuple=True)

    initial_state = cell.zero_state(1, data_type())

    input_list = []
    for _ in xrange(seed_size):
        d = tf.constant([2], data_type())
        input_list.append(d)

    input_data = tf.cast(input_list, tf.float32)
    input_data = tf.reshape(input_data, [1,1,seed_size])

    with tf.Session() as session:

        saver = tf.train.import_meta_graph(FLAGS.model_path)
        saver.restore(session, tf.train.latest_checkpoint('./'))

        graph = tf.get_default_graph()
        softmax_w = graph.get_tensor_by_name('Model/softmax_w:0')
        softmax_b = graph.get_tensor_by_name('Model/softmax_b:0')

        with tf.variable_scope("Model", reuse=None):
            inputs = tf.unstack(input_data, num=seed_size, axis=2)

            outputs, state = tf.contrib.rnn.static_rnn(
                cell, inputs, initial_state=initial_state)

            output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, hidden_size])

        logits = tf.matmul(output, softmax_w) + softmax_b

        init = tf.global_variables_initializer()
        session.run(init)

        fix = np.argmax(session.run(logits[-1,:]))

        for s in xrange(max_seq_length):
            print("Fix: " + str(fix))
            # if fix == 3:
            #     break

            d = tf.constant([fix], data_type())
            input_list.append(d)
            input_data = tf.cast(input_list, tf.float32)
            input_data = tf.reshape(input_data, [1,1,seed_size+s+1])

            with tf.variable_scope("Model", reuse=True):
                inputs = tf.unstack(input_data, num=seed_size+s+1, axis=2)

                outputs, state = tf.contrib.rnn.static_rnn(
                    cell, inputs, initial_state=state)

                output = tf.reshape(tf.stack(axis=1, values=outputs),
                                    [-1, hidden_size])

            logits = tf.matmul(output, softmax_w) + softmax_b
            fix = np.argmax(session.run(logits[-1,:]))


if __name__ == "__main__":
    tf.app.run()


