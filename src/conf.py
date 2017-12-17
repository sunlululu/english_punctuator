def get_config():
    return ModelConf()

class ModelConf(object):
    init_scale = 0.1 # scale to initialize LSTM weights
    learning_rate = 0.001
    max_grad_norm = 5
    embedding_size = 200
    hidden_size = 200
    num_proj = 200
    max_max_epoch = 10
    keep_prob = 1.0
    lr_decay = 0.9
    lr_decay_step = 1000
    batch_size = 128
    vocab_size = 15000 + 2
    punc_size = 7
