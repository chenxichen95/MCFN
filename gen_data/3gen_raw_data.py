import pandas as pd
import os
from tqdm import tqdm
from data_pro_utils import *

root_path = os.path.abspath('.')


def check_csv_null(filepath, file_class=None):
    data_csv = pd.read_csv(filepath)
    result = data_csv.isnull().sum()
    if file_class == 'question':
        print('question_id 空值个数：{}'.format(result[0]))
        print('content 空值个数:{}'.format(result[1]))
    elif file_class == 'answer':
        print('ans_id 空值个数：{}'.format(result[0]))
        print('question_id 空值个数：{}'.format(result[1]))
        print('content 空值个数:{}'.format(result[2]))
    elif file_class == 'train_candidates':
        print('question_id 空值个数：{}'.format(result[0]))
        print('pos_ans_id 空值个数：{}'.format(result[1]))
        print('neg_ans_id 空值个数:{}'.format(result[2]))
    elif file_class == 'dev_candidates':
        print('question_id 空值个数：{}'.format(result[0]))
        print('ans_id 空值个数：{}'.format(result[1]))
        print('cnt 空值个数:{}'.format(result[2]))
        print('label 空值个数:{}'.format(result[3]))
    elif file_class == 'test_candidates':
        print('question_id 空值个数：{}'.format(result[0]))
        print('ans_id 空值个数：{}'.format(result[1]))
        print('cnt 空值个数:{}'.format(result[2]))
        print('label 空值个数:{}'.format(result[3]))


def load_txt_data(filepath, file_class=None):
    txt_data = []
    with tqdm(total=get_lc(filepath)) as pbar:
        with open(filepath, 'r', encoding='utf-8') as fp:
            for index, line in enumerate(fp):
                if index == 0:
                    continue
                if file_class == 'train_candidates':
                    pbar.set_description('train_candidates: ')
                    question_id, pos_ans_id, neg_ans_id = line.strip().split(',')
                    txt_data.append(
                        RawExample(
                            question_id=question_id,
                            pos_ans_id=pos_ans_id,
                            neg_ans_id=neg_ans_id,
                        )
                    )
                    pbar.update(1)
                elif file_class == 'dev_candidates':
                    pbar.set_description('dev_candidates')
                    question_id, ans_id, cnt, label = line.strip().split(',')
                    txt_data.append(
                        RawExample(
                            question_id=question_id,
                            ans_id=ans_id,
                            cnt=cnt,
                            label=label,
                        )
                    )
                    pbar.update(1)
                elif file_class == 'test_candidates':
                    pbar.set_description('test_candidates')
                    question_id, ans_id, cnt, label = line.strip().split(',')
                    txt_data.append(
                        RawExample(
                            question_id=question_id,
                            ans_id=ans_id,
                            cnt=cnt,
                            label=label
                        )
                    )
                    pbar.update(1)
                else:
                    raise Exception('file_class is error')

    return txt_data


def load_csv_data(filepath, file_class=None):
    data_csv = pd.read_csv(filepath)
    data_count = data_csv.shape[0]
    data = []
    with tqdm(total=data_count) as pbar:
        for index in range(data_count):
            if file_class == 'question':
                pbar.set_description('question_csv')
                data.append(
                    RawExample(
                        question_id=str(data_csv.iloc[index, 0]),
                        content=data_csv.iloc[index, 1]
                    )
                )
                pbar.update(1)
            elif file_class == 'answer':
                pbar.set_description('answer_csv')
                data.append(
                    RawExample(
                        ans_id=str(data_csv.iloc[index, 0]),
                        question_id=str(data_csv.iloc[index, 1]),
                        content=data_csv.iloc[index, 2],
                    )
                )
                pbar.update(1)
    return data


def gen_raw_data(reload=True):
    raw_data_path = {
        'train_candidates': f'{raw_data_dir}/train_candidates.pkl',
        'dev_candidates': f'{raw_data_dir}/dev_candidates.pkl',
        'test_candidates': f'{raw_data_dir}/test_candidates.pkl',
        'question': f'{raw_data_dir}/question.pkl',
        'answer': f'{raw_data_dir}/answer.pkl',
    }

    # gen answer csv
    answer_csv = pd.read_csv(answer_csv_path)
    question_csv = pd.read_csv(question_csv_path)

    if reload:
        # get question raw data
        if os.path.exists(raw_data_path['question']):
            question = load_pkl_data(raw_data_path['question'])
        else:
            question = load_csv_data(question_csv_path, file_class='question')
            save_pkl_data(question, raw_data_path['question'])
        # get answer raw data
        if os.path.exists(raw_data_path['answer']):
            answer = load_pkl_data(raw_data_path['answer'])
        else:
            answer = load_csv_data(answer_csv_path, file_class='answer')
            save_pkl_data(answer, raw_data_path['answer'])
        # get train_candidates raw data
        if os.path.exists(raw_data_path['train_candidates']):
            train_candidates = load_pkl_data(raw_data_path['train_candidates'])
        else:
            train_candidates = load_txt_data(train_candidates_path, file_class='train_candidates')
            save_pkl_data(train_candidates, raw_data_path['train_candidates'])
        # get dev_candidates raw data
        if os.path.exists(raw_data_path['dev_candidates']):
            dev_candidates = load_pkl_data(raw_data_path['dev_candidates'])
        else:
            dev_candidates = load_txt_data(dev_candidates_path, file_class='dev_candidates')
            save_pkl_data(dev_candidates, raw_data_path['dev_candidates'])
        # get test_candidates raw data
        if os.path.exists(raw_data_path['test_candidates']):
            test_candidates = load_pkl_data(raw_data_path['test_candidates'])
        else:
            test_candidates = load_txt_data(test_candidates_path, file_class='test_candidates')
            save_pkl_data(test_candidates, raw_data_path['test_candidates'])

    else:
        # get question raw data
        question = load_csv_data(question_csv_path, file_class='question')
        save_pkl_data(question, raw_data_path['question'])
        # get answer raw data
        answer = load_csv_data(answer_csv_path, file_class='answer')
        save_pkl_data(answer, raw_data_path['answer'])
        # get train_candidates raw data
        train_candidates = load_txt_data(train_candidates_path, file_class='train_candidates')
        save_pkl_data(train_candidates, raw_data_path['train_candidates'])
        # get dev_candidates raw data
        dev_candidates = load_txt_data(dev_candidates_path, file_class='dev_candidates')
        save_pkl_data(dev_candidates, raw_data_path['dev_candidates'])
        # get test_candidates raw data
        test_candidates = load_txt_data(test_candidates_path, file_class='test_candidates')
        save_pkl_data(test_candidates, raw_data_path['test_candidates'])

    print('finish generating raw data ')


if __name__ == '__main__':
    answer_csv_path = f'{root_path}/data/answer.csv'
    question_csv_path = f'{root_path}/data/question.csv'
    train_candidates_path = f'{root_path}/data/train_candidates.txt'
    dev_candidates_path = f'{root_path}/data/dev_candidates.txt'
    test_candidates_path = f'{root_path}/data/test_candidates.txt'

    raw_data_dir = f'{root_path}/raw_data'
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)

    if 'test file exsit':
        print('test file exsit')
        print(f'{answer_csv_path} ==> {os.path.exists(answer_csv_path)}')
        print(f'{question_csv_path} ==> {os.path.exists(question_csv_path)}')
        print(f'{dev_candidates_path} ==> {os.path.exists(dev_candidates_path)}')
        print(f'{test_candidates_path} ==> {os.path.exists(test_candidates_path)}')
        print(f'{train_candidates_path} ==> {os.path.exists(train_candidates_path)}')

    if 'check_csv_null':
        print('check_csv_null')
        print(f'file:{question_csv_path}')
        check_csv_null(question_csv_path, file_class='question')
        print(f'file:{answer_csv_path}')
        check_csv_null(answer_csv_path, file_class='answer')
        print(f'file:{train_candidates_path}')
        check_csv_null(train_candidates_path, file_class='train_candidates')
        print(f'file:{dev_candidates_path}')
        check_csv_null(dev_candidates_path, file_class='dev_candidates')
        print(f'file:{test_candidates_path}')
        check_csv_null(test_candidates_path, file_class='test_candidates')

    if 'gen_raw_data':
        gen_raw_data(reload=True)


