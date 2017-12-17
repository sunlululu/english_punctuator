import collections
import os
import pickle
import sys

import numpy as np
import tensorflow as tf

from emoji import UNICODE_EMOJI

flags = tf.app.flags
flags.DEFINE_string("data_dir", "../punc_data",
                    """Directory where exists text dataset""")
flags.DEFINE_string("out_dir", "data",
                    """data | sentence_data""")
FLAGS = flags.FLAGS

MAX_SEQ_LEN = 100

class Conf(object):
    vocabulary_file = "word_dict" # relative path to raw_data_path
    punct_vocab_file = "punc_list" # relative path to raw_data_path
    train_data = "train.txt" # relative path to raw_data_path
    valid_data = "valid.txt" # relative path to raw_data_path
    test_data = "test.txt" # relative path to raw_data_path

def contains_emoji(input_str):
    for c in input_str:
        if c in UNICODE_EMOJI:
            return True
    return False

def get_punctuations(punct_vocab_file):
    with open(punct_vocab_file, 'r', encoding='utf8') as f:
        punctuations = {w.strip('\n'): i for (i, w) in enumerate(f)}
    return punctuations

def input_word_index(vocabulary, input_word):
    if input_word.isdigit():
        input_word = "<num>"
    if input_word in vocabulary:
        return vocabulary.get(input_word)
    elif contains_emoji(input_word):
        return vocabulary.get("emoji")
    return vocabulary.get(input_word, vocabulary["<unk>"])

def punctuation_index(punctuations, punctuation):
    return punctuations[punctuation]

def load_vocabulary(file_path):
    with open(file_path, 'r', encoding='utf8') as vocab:
        vocabulary = {w.strip(): i for (i, w) in enumerate(vocab)}
    if "<unk>" not in vocabulary:
        vocabulary["<unk>"] = len(vocabulary)
    if "<END>" not in vocabulary:
        vocabulary["<END>"] = len(vocabulary)
    return vocabulary

def sentences_to_ids(file_path, vocabulary, punctuations):
    inputs = []
    outputs = []
    
    with open(file_path, 'r', encoding='utf-8') as corpus:
        for line in corpus:
            # Skip blank line
            if len(line.strip()) == 0:
                continue
            punctuation = " "
            inputs.append([])
            outputs.append([])
            meet_first_word = False
            input_words = line.lower().split()
            prev_punc = -1
            i = 0
            while i < len(input_words):
                token = input_words[i]
                if token in punctuations and meet_first_word:
                    punctuation = token
                    if punctuation != " ":
                        prev_punc = i
                else:
                    meet_first_word = True
                    inputs[-1].append(input_word_index(vocabulary, token))
                    outputs[-1].append(punctuation_index(punctuations, punctuation))
                    punctuation = " "
                i = i + 1
                if len(inputs[-1]) == MAX_SEQ_LEN:
                    if (i - prev_punc) <= (MAX_SEQ_LEN / 2):
                        i = prev_punc + 1
                    else:
                        del inputs[-1]
                        del outputs[-1]
                    punctuation = " "
                    meet_first_word = False
                    inputs.append([])
                    outputs.append([])

            if len(inputs[-1]) <= 1 or len(inputs[-1]) >= MAX_SEQ_LEN:
                #print("del\n", inputs[-1], '\n', outputs[-1])
                del inputs[-1]
                del outputs[-1]
                continue
            inputs[-1].append(input_word_index(vocabulary, "<END>"))
            outputs[-1].append(punctuation_index(punctuations, punctuation))
    inputs = [ e for e in inputs if len(e) <= MAX_SEQ_LEN]
    outputs = [ e for e in outputs if len(e) <= MAX_SEQ_LEN]
    return inputs, outputs


def save_to_pickle(inputs, outputs, vocabulary, punctuations, output_path):
    data = {"inputs": inputs, "outputs": outputs,
            "vocabulary": vocabulary, "punctuations": punctuations}
    
    with open(output_path+".pkl", 'wb') as output_file:
        pickle.dump(data, output_file, protocol=pickle.HIGHEST_PROTOCOL)

def save_to_pickle_blstm(inputs, outputs, vocabulary, punctuations, output_path):
    data = {"inputs": inputs, "outputs": outputs,
            "vocabulary": vocabulary, "punctuations": punctuations}
    
    with open(output_path+".pkl", 'wb') as output_file:
        pickle.dump(data, output_file, protocol=pickle.HIGHEST_PROTOCOL)

def convert_file_according_words(file_path, vocabulary, punctuations, output_path):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    print("Converting " + file_path)
    if tf.gfile.Exists(output_path):
        tf.gfile.DeleteRecursively(output_path)
    tf.gfile.MakeDirs(output_path)

    inputs, outputs = words_to_ids(file_path, vocabulary, punctuations)
    print("Length of inputs is " + str(len(inputs)))
    assert len(inputs) == len(outputs)

    save_to_pickle(inputs, outputs, vocabulary, punctuations, output_path)

    EXAMPLES_PER_FILE = 500000 #TODO: Make it an parameter
    NUM_FILES = int(np.floor(len(inputs)/EXAMPLES_PER_FILE))
    for i in range(NUM_FILES):
        filename = os.path.join(output_path,  "tfrecords-%.5d-of-%.5d" % (i+1, NUM_FILES))
        writer = tf.python_io.TFRecordWriter(filename)
        input = inputs[i*EXAMPLES_PER_FILE : (i+1)*EXAMPLES_PER_FILE]
        label = outputs[i*EXAMPLES_PER_FILE : (i+1)*EXAMPLES_PER_FILE]
        print("Writing " + filename + " with length of " + str(len(input)) + " data.")
        example = tf.train.Example(features=tf.train.Features(feature={
            "inputs": _int64_feature(input),
            "labels": _int64_feature(label)}))
        writer.write(example.SerializeToString())
        writer.close()
    print("Converting Successfully.")


def make_example(sequence, labels):
    ex = tf.train.SequenceExample()
    #print(len(sequence))
    ex.context.feature["length"].int64_list.value.append(len(sequence))
    fl_inputs = ex.feature_lists.feature_list["inputs"]
    fl_labels = ex.feature_lists.feature_list["labels"]
    for input, label in zip(sequence, labels):
        fl_inputs.feature.add().int64_list.value.append(input)
        fl_labels.feature.add().int64_list.value.append(label)
    return ex


def convert_file_according_sentences(file_path, vocabulary, punctuations, output_path):
    print("Converting " + file_path)
    if tf.gfile.Exists(output_path):
        tf.gfile.DeleteRecursively(output_path)
    tf.gfile.MakeDirs(output_path)

    inputs, outputs = sentences_to_ids(file_path, vocabulary, punctuations)
    inputs.sort(key=lambda x:len(x))
    outputs.sort(key=lambda x:len(x))
    print("Number of sentence is " + str(len(inputs)))
    assert len(inputs) == len(outputs)

    save_to_pickle_blstm(inputs, outputs, vocabulary, punctuations, output_path)

    SENTENCES_PER_FILE = 5000
    NUM_FILES = int(np.ceil(len(inputs)/SENTENCES_PER_FILE))
    for i in range(NUM_FILES):
        filename = os.path.join(output_path,  "tfrecords-%.5d-of-%.5d" % (i+1, NUM_FILES))
        writer = tf.python_io.TFRecordWriter(filename)
        seqs = inputs[i*SENTENCES_PER_FILE: (i+1)*SENTENCES_PER_FILE]
        labs = outputs[i*SENTENCES_PER_FILE: (i+1)*SENTENCES_PER_FILE]
        print("Writing " + filename + " with " + str(len(seqs)) + " sentences.")
        for seq, label_seq in zip(seqs, labs):
            ex = make_example(seq, label_seq)
            writer.write(ex.SerializeToString())
        writer.close()
    print("Converting Successfully.")


def convert_text_to_tfrecord(raw_data_path, conf, output_dir="data"):
    vocab_file = os.path.join(raw_data_path, conf.vocabulary_file)
    punct_vocab_file = os.path.join(raw_data_path, conf.punct_vocab_file)
    train_data = os.path.join(raw_data_path, conf.train_data)
    valid_data = os.path.join(raw_data_path, conf.valid_data)
    test_data = os.path.join(raw_data_path, conf.test_data)
    data_path = os.path.join(raw_data_path, output_dir)

    if not os.path.exists(vocab_file):
        print("Please build vocab by tools/generate_vocab.py")
        print(vocab_file)
        exit(0)
    vocabulary = load_vocabulary(vocab_file)
    punctuations = get_punctuations(punct_vocab_file)

    convert_file_according_sentences(train_data, vocabulary, punctuations,
                os.path.join(data_path, "train"))
    convert_file_according_sentences(valid_data, vocabulary, punctuations,
                os.path.join(data_path, "valid"))
    convert_file_according_sentences(test_data, vocabulary, punctuations,
                os.path.join(data_path, "test"))


def main():
    convert_text_to_tfrecord(FLAGS.data_dir, conf=Conf(), output_dir=FLAGS.out_dir)


if __name__ == "__main__":
    main()
