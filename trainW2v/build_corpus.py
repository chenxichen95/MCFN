from utils import *
from tqdm import tqdm
import os
if __name__ == '__main__':
    train_text_data_path = os.path.abspath('..') + '/gen_data/data/train_text_data.pkl'
    train_text_data = load_pkl_data(train_text_data_path)
    question_ids = {}
    pos_ans_ids = {}
    neg_ans_ids = {}
    corpus = []
    corpus_sentence = []
    corpus_path = os.path.abspath('.') + '/data/corpus.txt'
    corpus_sentence_path = os.path.abspath('.') + '/data/corpus_sentence.pkl'
    with tqdm(total=len(train_text_data)) as pbar:
        for item in train_text_data:
            question_id = item.question_id
            pos_ans_id = item.pos_ans_id
            neg_ans_id = item.neg_ans_id
            if question_ids.get(question_id, 'N') == 'N':
                question_ids[question_id] = 'Y'
                corpus.append(item.q_content)
            if pos_ans_ids.get(pos_ans_id, 'N') == 'N':
                pos_ans_ids[pos_ans_id] = 'Y'
                corpus.append(item.p_ans_content)
            if neg_ans_ids.get(neg_ans_id, 'N') == 'N':
                neg_ans_ids[neg_ans_id] = 'Y'
                corpus.append(item.n_ans_content)
            pbar.update(1)
    with open(corpus_path, 'w', encoding='utf-8') as fp:
        for item in corpus:
            text = ''.join([token+' ' for token in item])
            fp.write(text)
    with tqdm(total=len(corpus)) as pbar:
        for item in corpus:
            sentence = [token for token in item]
            corpus_sentence.append(sentence)
            pbar.update(1)
    save_pkl_data(corpus_sentence, corpus_sentence_path)
    print('end')