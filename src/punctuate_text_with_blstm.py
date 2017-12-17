import logging

import tensorflow as tf

import punc_input_blstm
import utils
from conf import *
import blstm
from convert_text_to_TFRecord import *
import re

flags = tf.flags

flags.DEFINE_string("input_file", None,
                    "Where the unpunctuated text is stored (the text is already segmented).")
flags.DEFINE_string("output_file", "./punctuated_text",
                    "Where the punctuated text you want to put.")
flags.DEFINE_string("vocabulary", "word_dict",
                    "The same vocabulary used to train the LSTM model.")
flags.DEFINE_string("punct_vocab", "punc_list",
                    "The same punctution vocabulary used to train the LSTM model.")
flags.DEFINE_string("save_path", "model_015/",
    "Model output directory.")
flags.DEFINE_string("log", "log",
    "Log filename.")
FLAGS = flags.FLAGS

logging.basicConfig(filename=FLAGS.log, filemode='w',
                    level=logging.INFO,
                    format='[%(levelname)s %(asctime)s] %(message)s')

def process_line(input_str):
    s = input_str
    s = re.sub('[0-9]+', ' <num> ', s)
    s = re.sub('<num> . <num>', ' <num> ', s)
    s = re.sub(r'([\.,!\?\:])', r' \1 ', s)
    s = re.sub(r'(\w)\'(\s|$)', r"\1 ' \2", s)
    s = re.sub(r'(\s|^)\'(\w)', r"\1 ' \2", s)
    s = re.sub('[ ]+', ' ', s).strip()
    return s

def inference_sentences_to_ids(file_path, vocabulary, punctuations):
    inputs = []
    outputs = []
    lens = []
    
    with open(file_path, 'r', encoding='utf-8') as corpus:
        for line in corpus:
            # Skip blank line
            if len(line.strip()) == 0:
                continue
            punctuation = " "
            inputs.append([])
            outputs.append([])
            line = process_line(line)
            for token in line.split():
                if token in punctuations:
                    punctuation = token
                else:
                    inputs[-1].append(input_word_index(vocabulary, token))
                    outputs[-1].append(punctuation_index(punctuations, punctuation))
                    punctuation = " "
            inputs[-1].append(input_word_index(vocabulary, "<END>"))
            outputs[-1].append(punctuation_index(punctuations, punctuation))
            lens.append(int(len(inputs[-1])))
    return inputs, outputs, lens


def get_predicts(inputs, outputs, lens):
    config = get_config()
    config.num_steps = 1
    config.batch_size = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(
            -config.init_scale, config.init_scale)

        # Generate LSTM batch
        input_batch, label_batch, len_batch = punc_input_blstm.eval_inputs(
            inputs=inputs,
            outputs=outputs,
            lens=lens)

        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            mtest = blstm.BLSTMPunctuator(input_batch=input_batch, label_batch=label_batch,
                              seq_len=len_batch, is_training=False, config=config)

        sv = tf.train.Supervisor()
        with sv.managed_session() as session:
            logging.info("Number of parameters: {}".format(utils.count_number_trainable_params()))

            ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
            if ckpt and ckpt.model_checkpoint_path:
                logging.info("Model checkpoint file path: " + ckpt.model_checkpoint_path)
                sv.saver.restore(session, ckpt.model_checkpoint_path)
            else:
                logging.info("No checkpoint file found")
                return

            epoch_size = len(inputs) #// config.batch_size

            test_perplexity, predicts = blstm.run_epoch(session, mtest, verbose=True, epoch_size=epoch_size)
            logging.info("Test Perplexity: %.3f" % test_perplexity)
        
        return predicts


def write_punctuations(input_file, predicts, punct_list, punct_vocab_reverse_map, output_file):
    with open(input_file, 'r', encoding='utf8') as inpf, open(output_file, 'w', encoding='utf8') as outf:
        i = 0
        for line in inpf:
            j = 0
            line = process_line(line)
            for word in line.split():
                if word in punct_list:
                    continue
                punctuation = punct_vocab_reverse_map[predicts[i][j]]
                if punctuation == " ":
                    outf.write("%s " % (word))
                else:
                    outf.write("%s %s " % (punctuation, word))
                j += 1
            # <END>
            punctuation = punct_vocab_reverse_map[predicts[i][j]]
            outf.write("%s\n" % (punctuation))
            i += 1

def evaluate_predict_result(predicts, labels):
    config = get_config()
    num_of_punc = config.punc_size

    mat = [[0 for j in range(num_of_punc)] for i in range(num_of_punc)]

    for i in range(len(predicts)):
        for j in range(len(predicts[i])):
            pred = predicts[i][j]
            labl = labels[i][j]
            mat[pred][labl] += 1
    return mat

def print_result(evaluate_res, punc_map):
    num_of_punc = len(evaluate_res)
    for res in evaluate_res:
        print("\t".join([str(i) for i in res]))

    for i in range(num_of_punc):
        predict_sum = sum(evaluate_res[i])
        precision = "-"
        if predict_sum != 0:
            precision = "%.3f" % (evaluate_res[i][i] * 1.0 / predict_sum)
        label_sum = sum([evaluate_res[x][i] for x in range(num_of_punc)])
        recall = "-"
        if label_sum != 0:
            recall = "%.3f" % (evaluate_res[i][i] * 1.0 / label_sum)
        print("Label %d - %s :  precision: %s  recall: %s" % (i, punc_map[i], precision, recall))

    sum_accu = sum([evaluate_res[i][i] for i in range(1, num_of_punc)])
    sum_all_pred = sum([sum(evaluate_res[i]) for i in range(1, num_of_punc)])
    sum_all_labl = sum([sum(evaluate_res[i][1:]) for i in range(num_of_punc)])
    print("All :  precision: %.3f  recall: %.3f" % (sum_accu * 1.0 / sum_all_pred, sum_accu * 1.0 / sum_all_labl))


def punctuator(input_file, vocab_file, punct_vocab_file, output_file):
    # Convert text to ids. (NOTE: fake outputs)
    vocabulary = load_vocabulary(vocab_file)
    punctuations = get_punctuations(punct_vocab_file)
    punct_vocab_reverse_map = utils.get_reverse_map(punctuations)
    inputs, outputs, lens = inference_sentences_to_ids(input_file, vocabulary, punctuations)

    # Get predicts
    predicts = get_predicts(inputs, outputs, lens)

    # Write punctuations
    write_punctuations(input_file, predicts, punctuations, punct_vocab_reverse_map, output_file)
    predict_res = evaluate_predict_result(predicts, outputs)
    print_result(predict_res, punct_vocab_reverse_map)


if __name__ == "__main__":
    vocab_file = os.path.join(FLAGS.save_path, FLAGS.vocabulary)
    punct_file = os.path.join(FLAGS.save_path, FLAGS.punct_vocab)
    punctuator(FLAGS.input_file, vocab_file, punct_file, FLAGS.output_file)
