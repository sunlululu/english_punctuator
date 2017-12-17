import logging
import os

import numpy as np
import tensorflow as tf

import punc_input_blstm
import utils
import blstm
import conf

flags = tf.flags

flags.DEFINE_string("data_path", None,
    "Where the training/test data is stored."
)
flags.DEFINE_string("save_path", None,
    "Model output directory."
)
flags.DEFINE_string("log", "log",
    "Log filename."
)

FLAGS = flags.FLAGS

logging.basicConfig(filename=FLAGS.log, filemode='w',
                    level=logging.INFO,
                    format='[%(levelname)s %(asctime)s] %(message)s')


def train():
    """ Train Punctuator for a number of epochs."""
    config = conf.get_config()
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(
            -config.init_scale, config.init_scale)

        with tf.name_scope("Train"):
            train_input_batch, train_label_batch, train_seq_len, train_files = punc_input_blstm.inputs(os.path.join(FLAGS.data_path, "train"),
                                                            batch_size=config.batch_size)
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = blstm.BLSTMPunctuator(input_batch=train_input_batch, label_batch=train_label_batch,
                              seq_len=train_seq_len, is_training=True, config=config)
            tf.summary.scalar("Training_Loss", m.cost)
            tf.summary.scalar("Learning_Rate", m.lr)

        with tf.name_scope("Valid"):
            valid_input_batch, valid_label_batch, valid_seq_len, valid_files = punc_input_blstm.inputs(os.path.join(FLAGS.data_path, "valid"),
                                                            batch_size=config.batch_size)
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                m_valid = blstm.BLSTMPunctuator(input_batch=valid_input_batch, label_batch=valid_label_batch,
                              seq_len=valid_seq_len, is_training=False, config=config)
            tf.summary.scalar("Valid_Loss", m_valid.cost)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
            logging.info("Number of parameters: {}".format(utils.count_number_trainable_params()))
            logging.info(session.run(train_files))
            logging.info(session.run(valid_files))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)
            epoch_size = punc_input_blstm.get_epoch_size(FLAGS.data_path + "/train.pkl",
                                                   config.batch_size)
            valid_epoch_size = punc_input_blstm.get_epoch_size(FLAGS.data_path + "/valid.pkl",
                                        config.batch_size)

            for i in range(config.max_max_epoch):
                # m.assign_lr(session, learning_rate)
                logging.info("Epoch: %d Learning rate: %f" % (i + 1, session.run(m.lr)))
                train_perplexity = blstm.run_epoch(session, m, eval_op=m.train_op, verbose=True,
                                            epoch_size=epoch_size)
                if np.isnan(train_perplexity):
                    logging.info("Perplexity is nan! now exit")
                    break
                logging.info("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                
                valid_perplexity, _ = blstm.run_epoch(session, m_valid, verbose=True, epoch_size=valid_epoch_size, debug=False)
                logging.info("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

                logging.info("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path,
                          global_step=sv.global_step)

            coord.request_stop()
            coord.join(threads)

def main(argv=None):
    train()


if __name__ == "__main__":
    tf.app.run()
