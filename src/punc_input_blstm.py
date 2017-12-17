import os

import numpy as np
import tensorflow as tf

from convert_text_to_TFRecord import *

class FileConf(object):
    vocabulary_file = "word_dict" # relative path to raw_data_path
    punct_vocab_file = "punc_list" # relative path to raw_data_path
    train_data = "train.txt" # relative path to raw_data_path
    valid_data = "valid.txt" # relative path to raw_data_path
    test_data = "test.txt" # relative path to raw_data_path

def inputs(data_dir, batch_size=1, tfrecords_format="tfrecords-*", fileshuf=True):
    """Construct input and label for punctuation prediction.

    Args:
        data_dir:
        num_steps:
        batch_size:

    Returns:
        input_batch: tensor of [batch_size, num_steps] 
        label_batch: tensor of [batch_size, num_steps]
    """
    MATCH_FORMAT = os.path.join(data_dir, tfrecords_format)
    files = tf.train.match_filenames_once(MATCH_FORMAT)

    filename_queue = tf.train.string_input_producer(files, shuffle=fileshuf)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    context_features = {
        "length": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "inputs": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    seq_length, sequence = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )
    length = seq_length["length"]
    inputs = sequence["inputs"]
    labels = sequence["labels"]

    num_threads = 16
    capacity = 10000 + 20 * batch_size
    # TODO: try random shuffle batch
    batch = tf.train.batch(
        tensors=[inputs, labels, length],
        batch_size=batch_size,
        dynamic_pad=True,
        num_threads=num_threads,
        capacity=capacity
    )
    input_batch = batch[0]
    label_batch = batch[1]
    seq_len = batch[2]
    return input_batch, label_batch, seq_len, files

def eval_inputs(inputs, outputs, lens):
    N = len(inputs)
    max_len = max(lens)
    inputs_pad = np.zeros((N, max_len))
    labels_pad = np.zeros((N, max_len))
    for i, (input, output) in enumerate(zip(inputs, outputs)):
        end = lens[i]
        inputs_pad[i, :end] = input[:end]
        labels_pad[i, :end] = output[:end]

    inputs = tf.convert_to_tensor(inputs_pad, name="inputs", dtype=tf.int32)
    labels = tf.convert_to_tensor(labels_pad, name="labels", dtype=tf.int32)
    lens = tf.convert_to_tensor(lens, name="lens", dtype=tf.int32)

    epoch_size = N

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    input_batch = tf.reshape(inputs[i][:lens[i]], [1, -1])
    label_batch = tf.reshape(labels[i][:lens[i]], [1, -1])
    seq_len = tf.reshape(lens[i], [-1])
    return input_batch, label_batch, seq_len

def get_epoch_size(pickle_file, batch_size):
    data=np.load(pickle_file)
    data_len=len(data["inputs"])
    epoch_size = data_len // batch_size
    return epoch_size

def data_len(pickle_file):
    data=np.load(pickle_file)
    return len(data["inputs"])
