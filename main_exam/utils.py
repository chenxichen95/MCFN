import pickle
import os
from tqdm import tqdm
import sys
import torch
import numpy as np
import random
from torch.utils.data import TensorDataset
import pymongo

sys.path.append('/home/chenxichen/pycharm_remote/pycharm_py3.6torch/paper_one')

def save_pkl_data(data, filename):
    data_pkl = pickle.dumps(data)
    with open(filename, 'wb') as fp:
        fp.write(data_pkl)


def load_pkl_data(filename):
    with open(filename, 'rb') as fp:
        data_pkl = fp.read()
    return pickle.loads(data_pkl)


class TrainTextExample(object):
    def __init__(self, question_id, q_content, pos_ans_id, p_ans_content, neg_ans_id, n_ans_content):

        self.question_id = question_id
        self.q_content = q_content
        self.pos_ans_id = pos_ans_id
        self.p_ans_content = p_ans_content
        self.neg_ans_id = neg_ans_id
        self.n_ans_content = n_ans_content

    def __str__(self):
        self.__repr__()

    def __repr__(self):
        s = "{"
        s += f'question_id: {self.question_id}'

        s += f'||q_content: {self.q_content}\n'

        s += f'pos_ans_id: {self.pos_ans_id}'

        s += f'||p_ans_content: {self.p_ans_content}\n'

        s += f'neg_ans_id: {self.neg_ans_id}'

        s += f'||n_ans_content: {self.n_ans_content}'

        s += "}"

        return s


class TrainFeatures(object):
    """A single set of features of data."""
    def __init__(
            self,
            question_id,
            pos_ans_id,
            neg_ans_id,
            q_input_ids,
            q_input_mask,
            q_segment_ids,
            p_a_input_ids,
            p_a_input_mask,
            p_a_segment_ids,
            n_a_input_ids,
            n_a_input_mask,
            n_a_segment_ids,
    ):
        self.question_id = question_id
        self.pos_ans_id = pos_ans_id
        self.neg_ans_id = neg_ans_id
        self.q_input_ids = q_input_ids
        self.q_input_mask = q_input_mask
        self.q_segment_ids = q_segment_ids
        self.p_a_input_ids = p_a_input_ids
        self.p_a_input_mask = p_a_input_mask
        self.p_a_segment_ids = p_a_segment_ids
        self.n_a_input_ids = n_a_input_ids
        self.n_a_input_mask = n_a_input_mask
        self.n_a_segment_ids = n_a_segment_ids


class DevTextExample(object):
    def __init__(self, question_id, q_content, ans_id, a_content, cnt, label):
        self.question_id = question_id
        self.q_content = q_content
        self.ans_id = ans_id
        self.a_content = a_content
        self.cnt = cnt
        self.label = label

    def __str__(self):
        self.__repr__()

    def __repr__(self):
        s = "{"
        s += f'question_id: {self.question_id}'

        s += f'||q_content: {self.q_content}\n'

        s += f'ans_id: {self.ans_id}'

        s += f'||a_content: {self.a_content}\n'

        s += f'cnt: {self.cnt}'

        s += f'||label: {self.label}'

        s += "}"

        return s


class DevFeatures(object):
    """A single set of features of data."""
    def __init__(
            self,
            question_id,
            ans_id,
            q_input_ids,
            q_input_mask,
            q_segment_ids,
            a_input_ids,
            a_input_mask,
            a_segment_ids,
            cnt,
            label,
            q_w2v_ids=None,
            a_w2v_ids=None,
    ):
        self.question_id = question_id
        self.ans_id = ans_id
        self.q_input_ids = q_input_ids
        self.q_input_mask = q_input_mask
        self.q_segment_ids = q_segment_ids
        self.a_input_ids = a_input_ids
        self.a_input_mask = a_input_mask
        self.a_segment_ids = a_segment_ids
        self.cnt = cnt
        self.label = label
        self.q_w2v_ids = q_w2v_ids
        self.a_w2v_ids = a_w2v_ids


class TestFeatures(object):
    """A single set of features of data."""
    def __init__(
            self,
            question_id,
            ans_id,
            q_input_ids,
            q_input_mask,
            q_segment_ids,
            a_input_ids,
            a_input_mask,
            a_segment_ids,
            cnt,
            label,
            q_w2v_ids=None,
            a_w2v_ids=None,
    ):
        self.question_id = question_id
        self.ans_id = ans_id
        self.q_input_ids = q_input_ids
        self.q_input_mask = q_input_mask
        self.q_segment_ids = q_segment_ids
        self.a_input_ids = a_input_ids
        self.a_input_mask = a_input_mask
        self.a_segment_ids = a_segment_ids
        self.cnt = cnt
        self.label = label
        self.q_w2v_ids = q_w2v_ids
        self.a_w2v_ids = a_w2v_ids


class TestTextExample(object):
    def __init__(self, question_id, q_content, ans_id, a_content, cnt, label):
        self.question_id = question_id
        self.q_content = q_content
        self.ans_id = ans_id
        self.a_content = a_content
        self.cnt = cnt
        self.label = label

    def __str__(self):
        self.__repr__()

    def __repr__(self):
        s = "{"
        s += f'question_id: {self.question_id}'

        s += f'||q_content: {self.q_content}\n'

        s += f'ans_id: {self.ans_id}'

        s += f'||a_content: {self.a_content}\n'

        s += f'cnt: {self.cnt}'

        s += f'||label: {self.label}'

        s += "}"

        return s


def convert_bert_feature(content, tokenizer, max_seq_length):
    content_token = tokenizer.tokenize(content)
    if len(content_token) > (max_seq_length - 1):
        content_token = content_token[:max_seq_length - 1]
    content_token = ["[CLS]"] + content_token

    input_ids = tokenizer.convert_tokens_to_ids(content_token)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, input_mask, segment_ids


def convert_w2v_id(content, max_seq_length):
    content_ids = [w2v_vocab2id.get(token, w2v_vocab2id['UNK']) for token in content]
    if len(content_ids) > max_seq_length:
        content_ids = content_ids[:max_seq_length]
    padding = [0] * (max_seq_length - len(content_ids))
    content_ids += padding
    return content_ids


def convert_train_text_to_features(text_examples, tokenizer, max_seq_length):
    features = []
    with tqdm(total=len(text_examples)) as pbar:
        pbar.set_description('convert train text examples to features')
        for example in text_examples:
            question_id = example.question_id
            q_content = example.q_content
            pos_ans_id = example.pos_ans_id
            p_ans_content = example.p_ans_content
            neg_ans_id = example.neg_ans_id
            n_ans_content = example.n_ans_content

            q_input_ids, q_input_mask, q_segment_ids = convert_bert_feature(
                q_content, tokenizer, max_seq_length)
            p_a_input_ids, p_a_input_mask, p_a_segment_ids = convert_bert_feature(
                p_ans_content, tokenizer, max_seq_length)
            n_a_input_ids, n_a_input_mask, n_a_segment_ids = convert_bert_feature(
                n_ans_content, tokenizer, max_seq_length)
            features.append(
                TrainFeatures(
                    question_id=int(question_id),
                    pos_ans_id=int(pos_ans_id),
                    neg_ans_id=int(neg_ans_id),
                    q_input_ids=q_input_ids,
                    q_input_mask=q_input_mask,
                    q_segment_ids=q_segment_ids,
                    p_a_input_ids=p_a_input_ids,
                    p_a_input_mask=p_a_input_mask,
                    p_a_segment_ids=p_a_segment_ids,
                    n_a_input_ids=n_a_input_ids,
                    n_a_input_mask=n_a_input_mask,
                    n_a_segment_ids=n_a_segment_ids,
                )
            )
            pbar.update(1)

    return features


def convert_train_text_to_features_mongo(text_examples, tokenizer, max_seq_length):
    sample_index = []
    with tqdm(total=len(text_examples)) as pbar:
        pbar.set_description('convert train text examples to features')
        for index, example in enumerate(text_examples):
            question_id = example.question_id
            q_content = example.q_content
            pos_ans_id = example.pos_ans_id
            p_ans_content = example.p_ans_content
            neg_ans_id = example.neg_ans_id
            n_ans_content = example.n_ans_content

            q_input_ids, q_input_mask, q_segment_ids = convert_bert_feature(
                q_content, tokenizer, max_seq_length)
            p_a_input_ids, p_a_input_mask, p_a_segment_ids = convert_bert_feature(
                p_ans_content, tokenizer, max_seq_length)
            n_a_input_ids, n_a_input_mask, n_a_segment_ids = convert_bert_feature(
                n_ans_content, tokenizer, max_seq_length)

            data = {
                'index': index,
                'question_id': int(question_id),
                'pos_ans_id': int(pos_ans_id),
                'neg_ans_id': int(neg_ans_id),
                'q_input_ids': q_input_ids,
                'q_input_mask': q_input_mask,
                'q_segment_ids': q_segment_ids,
                'p_a_input_ids': p_a_input_ids,
                'p_a_input_mask': p_a_input_mask,
                'p_a_segment_ids': p_a_segment_ids,
                'n_a_input_ids': n_a_input_ids,
                'n_a_input_mask': n_a_input_mask,
                'n_a_segment_ids': n_a_segment_ids,
            }
            collection.insert(data)
            sample_index.append(index)
            pbar.update(1)

    return sample_index


def convert_train_text_to_features_mongo_w2v(text_examples, tokenizer, max_seq_length):
    sample_index = []
    with tqdm(total=len(text_examples)) as pbar:
        pbar.set_description('convert train text examples to features')
        for index, example in enumerate(text_examples):
            question_id = example.question_id
            q_content = example.q_content
            pos_ans_id = example.pos_ans_id
            p_ans_content = example.p_ans_content
            neg_ans_id = example.neg_ans_id
            n_ans_content = example.n_ans_content

            q_w2v_ids = convert_w2v_id(q_content, max_seq_length)
            p_a_w2v_ids = convert_w2v_id(p_ans_content, max_seq_length)
            n_a_w2v_ids = convert_w2v_id(n_ans_content, max_seq_length)

            q_input_ids, q_input_mask, q_segment_ids = convert_bert_feature(
                q_content, tokenizer, max_seq_length)
            p_a_input_ids, p_a_input_mask, p_a_segment_ids = convert_bert_feature(
                p_ans_content, tokenizer, max_seq_length)
            n_a_input_ids, n_a_input_mask, n_a_segment_ids = convert_bert_feature(
                n_ans_content, tokenizer, max_seq_length)

            data = {
                'index': index,
                'question_id': int(question_id),
                'pos_ans_id': int(pos_ans_id),
                'neg_ans_id': int(neg_ans_id),
                'q_input_ids': q_input_ids,
                'q_input_mask': q_input_mask,
                'q_segment_ids': q_segment_ids,
                'p_a_input_ids': p_a_input_ids,
                'p_a_input_mask': p_a_input_mask,
                'p_a_segment_ids': p_a_segment_ids,
                'n_a_input_ids': n_a_input_ids,
                'n_a_input_mask': n_a_input_mask,
                'n_a_segment_ids': n_a_segment_ids,
                'q_w2v_ids': q_w2v_ids,
                'p_a_w2v_ids': p_a_w2v_ids,
                'n_a_w2v_ids': n_a_w2v_ids,
            }
            collection.insert(data)
            sample_index.append(index)
            pbar.update(1)

    return sample_index


def convert_dev_text_to_features(text_examples, tokenizer, max_seq_length):
    features = []
    with tqdm(total=len(text_examples)) as pbar:
        pbar.set_description('convert dev text examples to features')
        for example in text_examples:

            question_id = example.question_id
            q_content = example.q_content
            ans_id = example.ans_id
            a_content = example.a_content
            cnt = example.cnt
            label = example.label

            q_input_ids, q_input_mask, q_segment_ids = convert_bert_feature(
                q_content, tokenizer, max_seq_length)
            a_input_ids, a_input_mask, a_segment_ids = convert_bert_feature(
                a_content, tokenizer, max_seq_length)

            features.append(
                DevFeatures(
                    question_id=int(question_id),
                    ans_id=int(ans_id),
                    q_input_ids=q_input_ids,
                    q_input_mask=q_input_mask,
                    q_segment_ids=q_segment_ids,
                    a_input_ids=a_input_ids,
                    a_input_mask=a_input_mask,
                    a_segment_ids=a_segment_ids,
                    cnt=int(cnt),
                    label=int(label),
                )
            )
            pbar.update(1)

    return features


def convert_dev_text_to_features_w2v(text_examples, tokenizer, max_seq_length):
    features = []
    with tqdm(total=len(text_examples)) as pbar:
        pbar.set_description('convert dev text examples to features')
        for example in text_examples:

            question_id = example.question_id
            q_content = example.q_content
            ans_id = example.ans_id
            a_content = example.a_content
            cnt = example.cnt
            label = example.label

            q_w2v_ids = convert_w2v_id(q_content, max_seq_length)
            a_w2v_ids = convert_w2v_id(a_content, max_seq_length)

            q_input_ids, q_input_mask, q_segment_ids = convert_bert_feature(
                q_content, tokenizer, max_seq_length)
            a_input_ids, a_input_mask, a_segment_ids = convert_bert_feature(
                a_content, tokenizer, max_seq_length)

            features.append(
                DevFeatures(
                    question_id=int(question_id),
                    ans_id=int(ans_id),
                    q_input_ids=q_input_ids,
                    q_input_mask=q_input_mask,
                    q_segment_ids=q_segment_ids,
                    a_input_ids=a_input_ids,
                    a_input_mask=a_input_mask,
                    a_segment_ids=a_segment_ids,
                    q_w2v_ids=q_w2v_ids,
                    a_w2v_ids=a_w2v_ids,
                    cnt=int(cnt),
                    label=int(label),
                )
            )
            pbar.update(1)

    return features


def convert_test_text_to_features(text_examples, tokenizer, max_seq_length):
    features = []
    with tqdm(total=len(text_examples)) as pbar:
        pbar.set_description('convert test text examples to features')
        for example in text_examples:

            question_id = example.question_id
            q_content = example.q_content
            ans_id = example.ans_id
            a_content = example.a_content
            cnt = example.cnt
            label = example.label

            q_input_ids, q_input_mask, q_segment_ids = convert_bert_feature(
                q_content, tokenizer, max_seq_length)
            a_input_ids, a_input_mask, a_segment_ids = convert_bert_feature(
                a_content, tokenizer, max_seq_length)

            features.append(
                TestFeatures(
                    question_id=int(question_id),
                    ans_id=int(ans_id),
                    q_input_ids=q_input_ids,
                    q_input_mask=q_input_mask,
                    q_segment_ids=q_segment_ids,
                    a_input_ids=a_input_ids,
                    a_input_mask=a_input_mask,
                    a_segment_ids=a_segment_ids,
                    cnt=int(cnt),
                    label=int(label),
                )
            )
            pbar.update(1)

    return features


def convert_test_text_to_features_w2v(text_examples, tokenizer, max_seq_length):
    features = []
    with tqdm(total=len(text_examples)) as pbar:
        pbar.set_description('convert test text examples to features')
        for example in text_examples:

            question_id = example.question_id
            q_content = example.q_content
            ans_id = example.ans_id
            a_content = example.a_content
            cnt = example.cnt
            label = example.label

            q_w2v_ids = convert_w2v_id(q_content, max_seq_length)
            a_w2v_ids = convert_w2v_id(a_content, max_seq_length)

            q_input_ids, q_input_mask, q_segment_ids = convert_bert_feature(
                q_content, tokenizer, max_seq_length)
            a_input_ids, a_input_mask, a_segment_ids = convert_bert_feature(
                a_content, tokenizer, max_seq_length)

            features.append(
                TestFeatures(
                    question_id=int(question_id),
                    ans_id=int(ans_id),
                    q_input_ids=q_input_ids,
                    q_input_mask=q_input_mask,
                    q_segment_ids=q_segment_ids,
                    a_input_ids=a_input_ids,
                    a_input_mask=a_input_mask,
                    a_segment_ids=a_segment_ids,
                    cnt=int(cnt),
                    label=int(label),
                    q_w2v_ids=q_w2v_ids,
                    a_w2v_ids=a_w2v_ids,
                )
            )
            pbar.update(1)

    return features


def get_data(args, file_class, save_mode='local', use_w2v=False):
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    # load features
    features = ''
    if file_class == 'train':
        if save_mode == 'local':
            save_dir = args.train_text_data.split('/')
            save_dir[-1] = f'train_features_data_{args.max_seq_length}'
            save_dir = "/".join(save_dir)
            if os.path.exists(save_dir):
                features = load_pkl_data(save_dir)
            else:
                text_examples = load_pkl_data(args.train_text_data)
                features = convert_train_text_to_features(text_examples, tokenizer, args.max_seq_length)
                save_pkl_data(features, save_dir)

        elif save_mode == 'mongo':
            save_dir = args.train_text_data.split('/')
            if use_w2v:
                save_dir[-1] = f'train_sample_index_{args.max_seq_length}_use_w2v'
            else:
                save_dir[-1] = f'train_sample_index_{args.max_seq_length}'
            save_dir = "/".join(save_dir)
            if os.path.exists(save_dir):
                train_sample_index = load_pkl_data(save_dir)
            else:
                text_examples = load_pkl_data(args.train_text_data)
                if use_w2v:
                    train_sample_index = convert_train_text_to_features_mongo_w2v(text_examples, tokenizer, args.max_seq_length)
                else:
                    train_sample_index = convert_train_text_to_features_mongo(text_examples, tokenizer, args.max_seq_length)
                save_pkl_data(train_sample_index, save_dir)
            return train_sample_index

    elif file_class == 'dev':
        save_dir = args.dev_text_data.split('/')
        if use_w2v:
            save_dir[-1] = f'dev_features_data_{args.max_seq_length}_use_w2v'
        else:
            save_dir[-1] = f'dev_features_data_{args.max_seq_length}'
        save_dir = "/".join(save_dir)
        if os.path.exists(save_dir):
            features = load_pkl_data(save_dir)
        else:
            text_examples = load_pkl_data(args.dev_text_data)
            if use_w2v:
                features = convert_dev_text_to_features_w2v(text_examples, tokenizer, args.max_seq_length)
            else:
                features = convert_dev_text_to_features(text_examples, tokenizer, args.max_seq_length)
            save_pkl_data(features, save_dir)

    elif file_class == 'test':
        save_dir = args.test_text_data.split('/')
        if use_w2v:
            save_dir[-1] = f'test_features_data_{args.max_seq_length}_use_w2v'
        else:
            save_dir[-1] = f'test_features_data_{args.max_seq_length}'
        save_dir = "/".join(save_dir)
        if os.path.exists(save_dir):
            features = load_pkl_data(save_dir)
        else:
            text_examples = load_pkl_data(args.test_text_data)
            if use_w2v:
                features = convert_test_text_to_features_w2v(text_examples, tokenizer, args.max_seq_length)
            else:
                features = convert_test_text_to_features(text_examples, tokenizer, args.max_seq_length)
            save_pkl_data(features, save_dir)

    return features


def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def get_tensor_dataset(args, file_class, save_mode='local', reload=True, use_w2v=False):
    tensor_dataset = ''
    if file_class == 'train':
        if save_mode == 'local':
            save_dir = args.train_text_data.split('/')
            save_dir[-1] = f'train_features_tensor_dataset_{args.max_seq_length}'
            save_dir = "/".join(save_dir)
            if os.path.exists(save_dir):
                tensor_dataset = load_pkl_data(save_dir)
            else:
                print('building train features tensor dataset')
                print("get_data(args, file_class='train')")
                train_features = get_data(args, file_class='train')
                print('building tensor')
                question_id_tensor = torch.tensor([feature.question_id for feature in train_features])
                pos_ans_id_tensor = torch.tensor([feature.pos_ans_id for feature in train_features])
                neg_ans_id_tensor = torch.tensor([feature.neg_ans_id for feature in train_features])
                q_input_ids_tensor = torch.tensor([feature.q_input_ids for feature in train_features])
                q_input_mask_tensor = torch.tensor([feature.q_input_mask for feature in train_features])
                q_segment_ids_tensor = torch.tensor([feature.q_segment_ids for feature in train_features])
                p_a_input_ids_tensor = torch.tensor([feature.p_a_input_ids for feature in train_features])
                p_a_input_mask_tensor = torch.tensor([feature.p_a_input_mask for feature in train_features])
                p_a_segment_ids_tensor = torch.tensor([feature.p_a_segment_ids for feature in train_features])
                n_a_input_ids_tensor = torch.tensor([feature.n_a_input_ids for feature in train_features])
                n_a_input_mask_tensor = torch.tensor([feature.n_a_input_mask for feature in train_features])
                n_a_segment_ids_tensor = torch.tensor([feature.n_a_segment_ids for feature in train_features])
                tensor_dataset = TensorDataset(
                    question_id_tensor,
                    pos_ans_id_tensor,
                    neg_ans_id_tensor,
                    q_input_ids_tensor,
                    q_input_mask_tensor,
                    q_segment_ids_tensor,
                    p_a_input_ids_tensor,
                    p_a_input_mask_tensor,
                    p_a_segment_ids_tensor,
                    n_a_input_ids_tensor,
                    n_a_input_mask_tensor,
                    n_a_segment_ids_tensor,
                )
                print('save data')
                save_pkl_data(tensor_dataset, save_dir)
        elif save_mode == 'mongo':
            save_dir = args.train_text_data.split('/')
            if use_w2v:
                save_dir[-1] = f'train_sample_index_dataset_{args.max_seq_length}_use_w2v'
            else:
                save_dir[-1] = f'train_sample_index_dataset_{args.max_seq_length}'
            save_dir = "/".join(save_dir)
            if os.path.exists(save_dir):
                tensor_dataset = load_pkl_data(save_dir)
            else:
                print('building train features tensor dataset')
                print("get_data(args, file_class='train')")
                train_sample_index = get_data(args, file_class='train', save_mode=save_mode, use_w2v=use_w2v)
                sample_index_tensor = torch.tensor([sample_index for sample_index in train_sample_index])
                tensor_dataset = TensorDataset(sample_index_tensor)
                save_pkl_data(tensor_dataset, save_dir)


    elif file_class == 'dev':
        save_dir = args.dev_text_data.split('/')
        if use_w2v:
            save_dir[-1] = f'dev_features_tensor_dataset_{args.max_seq_length}_use_w2v'
        else:
            save_dir[-1] = f'dev_features_tensor_dataset_{args.max_seq_length}'
        save_dir = "/".join(save_dir)
        if os.path.exists(save_dir) and reload:
            tensor_dataset = load_pkl_data(save_dir)
        else:
            print('building dev features tensor dataset')
            dev_features = get_data(args, file_class='dev', use_w2v=use_w2v)
            question_id_tensor = torch.tensor([feature.question_id for feature in dev_features])
            ans_id_tensor = torch.tensor([feature.ans_id for feature in dev_features])
            q_input_ids_tensor = torch.tensor([feature.q_input_ids for feature in dev_features])
            q_input_mask_tensor = torch.tensor([feature.q_input_mask for feature in dev_features])
            q_segment_ids_tensor = torch.tensor([feature.q_segment_ids for feature in dev_features])
            a_input_ids_tensor = torch.tensor([feature.a_input_ids for feature in dev_features])
            a_input_mask_tensor = torch.tensor([feature.a_input_mask for feature in dev_features])
            a_segment_ids_tensor = torch.tensor([feature.a_segment_ids for feature in dev_features])
            cnt_tensor = torch.tensor([feature.cnt for feature in dev_features])
            label_tensor = torch.tensor([feature.label for feature in dev_features])
            q_w2v_ids_tensor = torch.tensor([feature.q_w2v_ids for feature in dev_features])
            a_w2v_ids_tensor = torch.tensor([feature.a_w2v_ids for feature in dev_features])
            tensor_dataset = TensorDataset(
                question_id_tensor,
                ans_id_tensor,
                q_input_ids_tensor,
                q_input_mask_tensor,
                q_segment_ids_tensor,
                a_input_ids_tensor,
                a_input_mask_tensor,
                a_segment_ids_tensor,
                cnt_tensor,
                label_tensor,
                q_w2v_ids_tensor,
                a_w2v_ids_tensor,
            )
            save_pkl_data(tensor_dataset, save_dir)

    elif file_class == 'test':
        save_dir = args.test_text_data.split('/')
        if use_w2v:
            save_dir[-1] = f'test_features_tensor_dataset_{args.max_seq_length}_use_w2v'
        else:
            save_dir[-1] = f'test_features_tensor_dataset_{args.max_seq_length}'
        save_dir = "/".join(save_dir)
        if os.path.exists(save_dir):
            tensor_dataset = load_pkl_data(save_dir)
        else:
            print('building test features tensor dataset')
            test_features = get_data(args, file_class='test', use_w2v=use_w2v)
            question_id_tensor = torch.tensor([feature.question_id for feature in test_features])
            ans_id_tensor = torch.tensor([feature.ans_id for feature in test_features])
            q_input_ids_tensor = torch.tensor([feature.q_input_ids for feature in test_features])
            q_input_mask_tensor = torch.tensor([feature.q_input_mask for feature in test_features])
            q_segment_ids_tensor = torch.tensor([feature.q_segment_ids for feature in test_features])
            a_input_ids_tensor = torch.tensor([feature.a_input_ids for feature in test_features])
            a_input_mask_tensor = torch.tensor([feature.a_input_mask for feature in test_features])
            a_segment_ids_tensor = torch.tensor([feature.a_segment_ids for feature in test_features])
            cnt_tensor = torch.tensor([feature.cnt for feature in test_features])
            label_tensor = torch.tensor([feature.label for feature in test_features])
            q_w2v_ids_tensor = torch.tensor([feature.q_w2v_ids for feature in test_features])
            a_w2v_ids_tensor = torch.tensor([feature.a_w2v_ids for feature in test_features])
            tensor_dataset = TensorDataset(
                question_id_tensor,
                ans_id_tensor,
                q_input_ids_tensor,
                q_input_mask_tensor,
                q_segment_ids_tensor,
                a_input_ids_tensor,
                a_input_mask_tensor,
                a_segment_ids_tensor,
                cnt_tensor,
                label_tensor,
                q_w2v_ids_tensor,
                a_w2v_ids_tensor,
            )
            save_pkl_data(tensor_dataset, save_dir)

    return tensor_dataset


if __name__ == '__main__':
    from config import Config
    args = Config().args

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client['cmcqa_epilepsy']

    use_w2v = True
    if use_w2v:
        collection = db['train_use_w2v']
        w2v_vocab2id = load_pkl_data(args.w2v_vocab2id)
    else:
        collection = db['train']

    #print("get_data(args, file_class='train')")
    #_ = get_data(args, file_class='train')

    #print("get_data(args, file_class='dev')")
    #_ = get_data(args, file_class='dev')
    #print("get_data(args, file_class='test')")
    #_ = get_data(args, file_class='test')

    #print("get_tensor_dataset(args, file_class='train')")
    _ = get_tensor_dataset(args, file_class='train', save_mode='mongo', use_w2v=use_w2v)

    #print("get_tensor_dataset(args, file_class='dev')")
    _ = get_tensor_dataset(args, file_class='dev', reload=False, use_w2v=use_w2v)

    #print("get_tensor_dataset(args, file_class='test')")
    _ = get_tensor_dataset(args, file_class='test', reload=False, use_w2v=use_w2v)
    print('end')