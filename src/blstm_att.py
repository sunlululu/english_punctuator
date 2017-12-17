import inspect
import logging
import time

import numpy as np
import tensorflow as tf


def run_epoch(session, model, eval_op=None, verbose=False, epoch_size=1):
    """Runs the model on the given data."""
    start_time = time.time()
    all_words = 0
    costs = 0.0
    predicts = []

    fetches = {
        "cost": model.cost,
        "mask": model.mask,
        "predict": model.predicts,
        "seqlen": model.seq_len,
        "loss": model.loss,
        "label": model.label,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op
    # if debug:
    #     fetches["inputs"] = model.Dinputs
    #     fetches["states"] = model.Dstates
    #     fetches["outputs"] = model.Doutput

    logging.info("Epoch size: %d" % epoch_size) 
    print_idx = 0
    for step in range(epoch_size):
        vals = session.run(fetches)
        cost = vals["cost"]
        mask = vals["mask"]
        predict = vals["predict"]
        label = vals["label"]
        np.set_printoptions(threshold=np.nan)
        if eval_op is None:
            
            # if step > 497:
            #     #for i in range(len(mask)):
            #     #    print(mask[i])
            #     print(np.sum(mask, axis=1))
            #     print(vals["seqlen"])
            mask = np.array(np.round(mask), dtype=np.int32)
            shape = mask.shape
            # if step > 10 and step < 20:
            #     print(predict)
            #     print(np.argmax(predict, 1))
            predict = np.reshape(np.argmax(predict, 1), shape).tolist()
            mask = np.sum(mask, axis=1).tolist()
            for i in range(shape[0]):
                predicts.append(predict[i][:mask[i]])
            #predicts.extend(np.argmax(predict, 1).tolist())
            # if debug:
            #     WIDTH = 10
            #     np.set_printoptions(threshold=np.nan)
            #     # print each layer's output
            #     print(np.array(vals["inputs"]).shape)
            #     for layer_c, layer_h in vals["states"][0]:
            #         print(np.array(layer_c).shape)
            #         print(np.array(layer_h).shape)
            #     print(np.array(vals["outputs"]).shape)
            #     print(np.array(vals["predict"]).shape)
            #     print("embedding output (x_t) :")
            #     print(vals["inputs"][0][0][:WIDTH])
            #     i = 1
            #     for layer_c, layer_h in vals["states"][0]:
            #         print("lstm layer %d cell output (c_t) :" % i)
            #         print(layer_c[0][:WIDTH])
            #         print("lstm layer %d projection output (m_t) :" % i)
            #         print(layer_h[0][:WIDTH])
            #         i += 1
            #     print("before softmax output: ")
            #     print(vals["outputs"][0][:WIDTH])
            #     print("softmax output (y_t) : ")
            #     print(vals["predict"][0][:WIDTH])
            # Keep in mind, when eval, num_steps=1, batch_size>=1
            # if get_post:
            #     for e in predict:
            #         posteriors.append(e.tolist())

        costs += cost
        words = np.sum(mask)
        all_words += words

        if epoch_size < 100:
            verbose = False

        if (step * 10 / epoch_size) > print_idx and eval_op is not None:
            print_idx = step * 10 / epoch_size + 1
            logging.info("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / step),
                    all_words / (time.time() - start_time)))
            predict = np.argmax(predict, 1)
            label_flat = np.reshape(label, [-1])
            all_label_equal = np.equal(predict, label_flat)
            not_space_label = np.not_equal(label_flat, np.zeros(np.shape(label_flat)))
            not_space_equal = all_label_equal * not_space_label
            not_space_label_count = np.sum(not_space_label)
            not_space_equal_count = np.sum(not_space_equal)
            none_space_accuracy = not_space_equal_count / not_space_label_count
            logging.info("not space label: %d" % not_space_label_count)
            logging.info("not space correct: %d" % not_space_equal_count)
            logging.info("not space accuracy: %.3f" % none_space_accuracy)
            logging.info("cost: %.3f" % (costs / step))
        if np.isnan(np.exp(costs / step)):
            print("perplexity is nan")
            print("cost: %f  step: %d" % (costs, step))
            return np.exp(costs / step)

    if eval_op is None:
        predict = np.reshape(predict, [-1])
        label_flat = np.reshape(label, [-1])
        all_label_equal = np.equal(predict, label_flat)
        not_space_label = np.not_equal(label_flat, np.zeros(np.shape(label_flat)))
        not_space_equal = all_label_equal * not_space_label
        not_space_label_count = np.sum(not_space_label)
        not_space_equal_count = np.sum(not_space_equal)
        none_space_accuracy = not_space_equal_count / not_space_label_count
        logging.info("not space label: %d" % not_space_label_count)
        logging.info("not space correct: %d" % not_space_equal_count)
        logging.info("not space accuracy: %.3f" % none_space_accuracy)
        logging.info("cost: %.3f" % (costs / step))
        return np.exp(costs / epoch_size), predicts
    # elif get_post:
    #     # Keep in mind, when get_post, num_steps=1, batch_size=1
    #     return np.exp(costs / iters), posteriors
    else:
        return np.exp(costs / epoch_size)


class BLSTMPunctuator(object):
    """BLSTM Punctuation Prediction Model."""

    def __init__(self, input_batch, label_batch, seq_len, is_training, config):
        self.batch_size = batch_size = config.batch_size

        embedding_size = config.embedding_size
        hidden_size = config.hidden_size
        num_proj = config.num_proj
        vocab_size = config.vocab_size
        punc_size = config.punc_size

        # Set LSTM cells
        def lstm_cell():
            return tf.contrib.rnn.LSTMCell(
                hidden_size, use_peepholes=True, num_proj=num_proj,
                forget_bias=0.0, state_is_tuple=True,
                reuse=tf.get_variable_scope().reuse)
        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob)
        
        cells_fw = attn_cell()
        cells_bw = attn_cell()

        # Embedding part
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, embedding_size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, input_batch)
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # Automatically reset state in each batch
        with tf.variable_scope("BRNN"):
            (outputs_fw, outputs_bw) , _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cells_fw, 
                cell_bw=cells_bw,
                inputs=inputs,
                sequence_length=seq_len,
                dtype=tf.float32)
              
        self.Dinputs = inputs
        self.Dstates = []
        self.label = label_batch

        if num_proj is not None:
            hidden_size = num_proj

        outputs = tf.concat([outputs_fw, outputs_bw], 2)
        output = tf.reshape(outputs, [-1, hidden_size*2])
        self.Doutput = output

        softmax_w = tf.get_variable(
            "softmax_w", [hidden_size*2, punc_size], dtype=tf.float32)
        softmax_b = tf.get_variable(
            "softmax_b", [punc_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        self._predicts = predict = tf.nn.softmax(logits)

        # Generate mask matrix to mask loss
        maxlen = tf.cast(tf.reduce_max(seq_len), tf.int32) # it can not work on type int64
        ones = tf.ones([maxlen, maxlen], dtype=tf.float32)
        low_triangular_ones = tf.matrix_band_part(ones, -1, 0)
        mask = tf.gather(low_triangular_ones, seq_len-1)
        self._mask = tf.cast(mask, tf.int32)
        self.seq_len = seq_len
        mask_flat = tf.reshape(mask, [-1])

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=tf.reshape(label_batch, [-1])
        )
        self.loss = mask_flat * loss

        label_flat = tf.cast(tf.reshape(label_batch, [-1]), tf.int32)
        not_space = tf.not_equal(label_flat, tf.zeros(tf.shape(label_flat), dtype=tf.int32))
        self.label_flat = label_flat
        self.not_space = not_space

        self._cost = cost = tf.reduce_sum(mask_flat * loss) / tf.reduce_sum(mask_flat)
        
        # masked_loss = (mask_flat + tf.cast(not_space, tf.float32) * 0) * loss
        # cost = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask_flat)

        if not is_training:
            return

        # self._lr = tf.Variable(0.0, trainable=False)
        global_step = tf.contrib.framework.get_or_create_global_step()
        # init_lr = tf.convert_to_tensor(config.learning_rate, dtype=tf.float32)
        self._lr = tf.train.exponential_decay(config.learning_rate, global_step,
                                        config.lr_decay_step, config.lr_decay, staircase=True)
        tvars = tf.trainable_variables()
        grads, _ = self._grads = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        # optimizer = tf.train.GradientDescentOptimizer(self._lr)
        optimizer = tf.train.AdamOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=global_step)

        # self._new_lr = tf.placeholder(
        #    tf.float32, shape=[], name="new_learning_rate")
        # self._lr_update = tf.assign(self._lr, self._new_lr)
        # End __init__ 

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def cost(self):
        return self._cost

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def predicts(self):
        return self._predicts

    @property
    def grads(self):
        return self._grads

    @property
    def mask(self):
        return self._mask
