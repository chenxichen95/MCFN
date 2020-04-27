import gensim
from gensim.models import Word2Vec
from utils import *
import logging
import os


def get_train_sentence():
    return load_pkl_data('./data/corpus_sentence.pkl')

def create_w2v_model():
    model_config = {
        'sg': 0,
        'size': 300,
        'window': 5,
        'workers': 10,
        'hs': 1,
        'seed': 42,
        'iter': 1000,
    }
    return Word2Vec(**model_config), model_config


def build_vocab_vector(model, config):
    # build vocab2ids
    vocab2id = {'<PAD>': 0, 'UNK': 1}
    for vocab in model.wv.vocab:
        vocab2id[vocab] = len(vocab2id)

    vocab_weight = [[0 for _ in range(config['size'])] for _ in range(len(vocab2id))]
    for vocab in model.wv.vocab:
        vector = model[vocab].tolist()
        id = vocab2id[vocab]
        vocab_weight[id] = vector

    return vocab2id, vocab_weight

def build_vocab2id(model, config):
    # build vocab2ids
    vocab2id = {'<PAD>': 0, 'UNK': 1}
    for vocab in model.wv.vocab:
        vocab2id[vocab] = len(vocab2id)

    path = f"./model/w2v_iter{config['iter']}_size{config['size']}"
    if not os.path.exists(path):
        os.makedirs(path)
    save_pkl_data(vocab2id, f'{path}/vocab2id.pkl')


def save_all(model, config, vocab2id, vocab_weight):
    path = f"./model/w2v_iter{config['iter']}_size{config['size']}"
    if not os.path.exists(path):
        os.makedirs(path)
    model.save(f'{path}/w2v.model')
    save_pkl_data(config, f'{path}/config.pkl')
    save_pkl_data(vocab2id, f'{path}/vocab2id.pkl')
    save_pkl_data(vocab_weight, f'{path}/vocab_weight.pkl')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
    corpus_sentence = get_train_sentence()
    model, config = create_w2v_model()
    model.build_vocab(sentences=corpus_sentence)
    build_vocab2id(model, config)
    model.train(corpus_sentence, total_examples=model.corpus_count, epochs=model.iter)
    vocab2id, vocab_weight = build_vocab_vector(model, config)
    save_all(model, config, vocab2id, vocab_weight)