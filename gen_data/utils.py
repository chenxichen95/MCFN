import pymongo
from tqdm import tqdm
import copy
import os
import pickle
import re
import jieba
import pandas as pd
import numpy as np


def getData(db, save_dir, restore=True, save=True):
    dataset = []
    if os.path.exists(save_dir) and restore:
        dataset = load_pkl_data(save_dir)
    else:
        print(f'{save_dir} does not exist')
        with tqdm(total=db.eval('db.stats()').get('objects')) as pbar:
            for curCollection in db.collection_names():
                pbar.set_description(f'current collection: {curCollection}')
                for item in db[curCollection].find():
                    if '癫痫' in item.get('illnessType', 'None'):
                        dataset.append(item)
                    pbar.update(1)
        if save:
            save_pkl_data(dataset, save_dir)
    return dataset


def delRepetition(dataset=None, save_dir=None, restore=True, save=True):
    '''
        根据问题文本是否重复，对数据集进行去重处理

        return illnessSet_unique, repetitionList
    '''
    if os.path.exists(save_dir) and restore:
        dataset_unique = load_pkl_data(save_dir)
    else:
        questionUnique = {}  # 使用 python 的 字典结构 加速检索过程
        dataset_unique = []
        repetitionNum = 0
        with tqdm(total=len(dataset)) as pbar:
            pbar.set_description('数据去重')
            for index, item in enumerate(dataset):
                if questionUnique.get(item['Q'], 0) == 0:
                    questionUnique.setdefault(item['Q'], 1)
                    dataset_unique.append(item)
                else:
                    repetitionNum += 1
                pbar.update(1)
        print(f'有 {repetitionNum} 个样本重复。剩下 {len(dataset_unique)} 个样本')
        if save:
            save_pkl_data(dataset_unique, save_dir)
    return dataset_unique


def text_clear(dataset=None, save_dir=None, restore=True, save=True):
    '''
        文本处理：
            (1) 删除文本中的乱码，无用的标点符号，多余的空格 只保留规定的字符
            (2) 常见的英文标点符号 替换为 对应的中文标点符号
            (3) 将文本中的 大写英文字母 转 小写英文字母
            (4) 删除答案文本中的模板文本
    '''
    if os.path.exists(save_dir) and restore:
        dataset = load_pkl_data(save_dir)
    else:
        dataset0 = copy.deepcopy(dataset)
        keepChar = u'[^a-zA-Z\u4e00-\u9fa5,.:;''""?|!\^%()-\[\]{}/\`~，。：；‘’”“？|！……%（）—【】{}·~～、]'
        usualChinaPunc = ['，', '。', '：', '；', '？', '！', '（', '）', '【', '】']
        usualEngPunc = [',', '.', ':', ';', '?', '!', '(', ')', '[', ']']
        with tqdm(total=len(dataset)) as pbar:
            pbar.set_description('文本处理')
            for index, item in enumerate(dataset):
                # (1)
                item['Q'] = re.sub(f'{keepChar}', '', item['Q'])
                item['Q_detailed'] = re.sub(f'{keepChar}', '', item['Q_detailed'])
                item['A1'] = re.sub(f'{keepChar}', '', item['A1'])

                # (2)
                for curEngIndex, curEngPunc in enumerate(usualEngPunc):
                    item['Q'] = re.sub(f'[{curEngPunc}]', f'{usualChinaPunc[curEngIndex]}', item['Q'])
                    item['Q_detailed'] = re.sub(f'[{curEngPunc}]', f'{usualChinaPunc[curEngIndex]}', item['Q_detailed'])
                    item['A1'] = re.sub(f'[{curEngPunc}]', f'{usualChinaPunc[curEngIndex]}', item['A1'])
                # (3)
                item['Q'] = item['Q'].lower()
                item['Q_detailed'] = item['Q_detailed'].lower()
                item['A1'] = item['A1'].lower()
                # (4)
                item['A1'] = re.sub('病情分析：|指导意见：|医生建议：', '', item['A1'])

                pbar.update(1)

        # 检查多少样本被处理
        subNum = 0
        subSample = []
        for index, item in enumerate(dataset0):
            if item['Q'] != dataset[index]['Q']:
                subNum += 1
                subSample.append({'old': item, 'new': dataset[index]})
                continue
            elif item['Q_detailed'] != dataset[index]['Q_detailed']:
                subNum += 1
                subSample.append({'old': item, 'new': dataset[index]})
                continue
            elif item['A1'] != dataset[index]['A1']:
                subNum += 1
                subSample.append({'old': item, 'new': dataset[index]})
                continue
            else:
                continue

        print(f'在“文本处理”的过程中，样本被处理的数量： {subNum}({subNum / len(dataset):.2%})')
        if save:
            save_pkl_data(dataset, save_dir)
    return dataset


def filterQ(dataset, save_dir, restore=True, save=True):
    '''
        筛选 数据集 中的问题，剔除不满足条件的数据
        筛选条件：
            问题文本中含有关键词
            问题文本不超过 2 句。

        return dataset_filter
    '''
    if os.path.exists(save_dir) and restore:
        dataset_filter = load_pkl_data(save_dir)
    else:
        dataset_filter = []
        # 选择 问题过滤 的 关键词
        filterKeyWords = ['吗', '什么', '怎么', '哪些', '呢', '怎么办', '如何', '是不是', '为什么',
                          '怎样', '请问', '怎么样', '多少', '怎么回事', '哪里', '好不好', '有没有',
                          '可不可以', '几年', '几天', '哪个', '多久', '是否', '有用吗']
        with tqdm(total=len(dataset)) as pbar:
            pbar.set_description('问题筛选')
            for index, item in enumerate(dataset):
                curQ = item['Q']
                curQ_cut = list(jieba.cut(curQ))
                QSentenceNum = re.split('[，。？！……]', curQ)[:-1]
                if len(QSentenceNum) < 3:
                    for curFilterWord in filterKeyWords:
                        if curFilterWord in curQ_cut:
                            dataset_filter.append(item)
                            break
                pbar.update(1)
        subNum = len(dataset_filter)
        print(f'在“问句筛选”的过程中，剩余样本个数： {subNum}({subNum / len(dataset):.2%})')
        if save:
            save_pkl_data(dataset_filter, save_dir)
    return dataset_filter


def statisticChar(dataset):
    '''
        统计 “Q”，“Q_detailed”和“A” 的中文字个数

        return len_Q, len_Q_detailed, len_A
    '''
    len_Q = []
    len_Q_detailed = []
    len_A = []
    with tqdm(total=len(dataset)) as pbar:
        for index, item in enumerate(dataset):
            curQ = item['Q']
            len_Q.append(len(curQ))
            curQ_detailed = item['Q_detailed']
            len_Q_detailed.append(len(curQ_detailed))
            curA = item['A1']
            len_A.append(len(curA))
            pbar.update(1)

    len_Q = pd.DataFrame(len_Q)
    len_Q_detailed = pd.DataFrame(len_Q_detailed)
    len_A = pd.DataFrame(len_A)

    print('=' * 20)
    print('“Q_”的字量统计情况如下：')
    print(len_Q.describe(percentiles=[0.5, 0.8, 0.9, 0.99, 0.995]))
    print('=' * 20)
    print('=' * 20)
    print('“A”的字量统计情况如下：')
    print(len_A.describe(percentiles=[0.5, 0.8, 0.9, 0.95, 0.97]))
    print('=' * 20)
    return len_Q, len_Q_detailed, len_A


def delToLongSample(dataset, save_dir, maxLen=300, restore=True, save=True):
    '''
        根据 maxLen , 丢弃文本过长的样本
    '''
    dataset_delToLong = []
    if os.path.exists(save_dir) and restore:
        dataset_delToLong = load_pkl_data(save_dir)
    else:
        with tqdm(total=len(dataset)) as pbar:
            for index, item in enumerate(dataset):
                curQ_Len = len(item['Q'])
                curA_Len = len(item['A1'])

                if curQ_Len <= maxLen:
                    if curA_Len <= maxLen:
                        dataset_delToLong.append(item)
                pbar.update(1)

        print(f'经过“文本长度处理”后，剩余样本个数：{len(dataset_delToLong)}({len(dataset_delToLong) / len(dataset):.2%})')
        if save:
            save_pkl_data(dataset_delToLong, save_dir)
    return dataset_delToLong


def final_check(dataset, save_dir, restore=True, save=True):
    '''
            检查 dataset 中各个字段是否存在 空值，如果有剔除
        '''
    goodDataset = []
    if os.path.exists(save_dir) and restore:
        goodDataset = load_pkl_data(save_dir)
    else:
        print('最终测试：')
        with tqdm(total=len(dataset)) as pbar:
            pbar.set_description('空值检测')
            for index, item in enumerate(dataset):
                if (item['Q'] == '') or (item['Q'] == []):
                    pbar.update(1)
                    continue
                elif (item['A1'] == '') or (item['A1'] == []):
                    pbar.update(1)
                    continue
                else:
                    goodDataset.append(item)
                    pbar.update(1)
        print(f'删除含有空值的样本个数：{len(dataset) - len(goodDataset)}，剩余样本个数：{len(goodDataset)}({len(goodDataset) / len(dataset):.2%})')
        goodDataset = delRepetition(dataset=goodDataset, save_dir='', restore=False, save=False)
        if save:
            save_pkl_data(goodDataset, save_dir)
    return goodDataset


def writeToDB(dataset, new_db):
    with tqdm(total=len(dataset)) as pbar:
        for index, item in enumerate(dataset):
            new_db['data'].insert(
                {
                    'illness_type': item['illnessType'],
                    'q': item['Q'],
                    'a': item['A1'],
                }
            )
            pbar.update(1)


def save_pkl_data(data, filename):
    data_pkl = pickle.dumps(data)
    print(f'save pkl: {filename}')
    with open(filename, 'wb') as fp:
        fp.write(data_pkl)


def load_pkl_data(filename):
    print(f'load pkl: {filename}')
    with open(filename, 'rb') as fp:
        data_pkl = fp.read()
    return pickle.loads(data_pkl)


def gen_question_answer_csv(db, save_dir, max_num):
    max_num = int(max_num)
    questions = []
    answers = []
    # gen question csv
    with tqdm(total=max_num) as pbar:
        for index, item in enumerate(db['data'].find()):
            if index == max_num:
                break
            q_content = item['q']
            a_content = item['a']
            qid = index + 1
            aid = index + 1
            questions.append([f'{qid}', q_content])
            answers.append([f'{aid}', f'{qid}', a_content])
            pbar.update(1)
    questions_csv = pd.DataFrame(questions, columns=['question_id', 'content'])
    answers_csv = pd.DataFrame(answers, columns=['ans_id', 'question_id', 'content'])
    # save csv
    questions_csv.to_csv(save_dir+'/question.csv', index=0)
    answers_csv.to_csv(save_dir+'/answer.csv', index=0)

class Seed():
    def __init__(self, init_seed):
        self.seed = init_seed
    def __call__(self):
        self.seed += 1
        return self.seed - 1

def gen_candidates(max_num, config):
    seed = Seed(config.seed)
    np.random.seed(seed())
    random_indexes = np.random.permutation(np.arange(1, max_num+1, dtype=np.int32))
    train_indexes = random_indexes[:int(max_num*(1 - config.dev_rate - config.test_rate))]
    dev_indexes = random_indexes[int(max_num*(1 - config.dev_rate - config.test_rate)): int(max_num*(1 - config.test_rate))]
    test_indexes = random_indexes[int(max_num*(1 - config.test_rate)):]

    def get_neg_index(pos_index, max_num, pair_num):
        while True:
            np.random.seed(seed())
            neg_index = np.random.randint(1, max_num + 1, (pair_num)).tolist()
            if pos_index not in neg_index:
                break
        return neg_index


    # gen train candidates
    train_candidates = []
    with tqdm(total=len(train_indexes)) as pbar:
        pbar.set_description('gen train candidates')
        for index in train_indexes:
            question_id = index
            pos_ans_id = index
            neg_index = get_neg_index(pos_ans_id, max_num, pair_num=config.train_pair_num)
            train_candidates.extend(
                [[f'{question_id}', f'{pos_ans_id}', f'{neg_index[i]}'] for i in range(config.train_pair_num)]
            )
            pbar.update(1)
    print(f'save {config.data_dir}/train_candidates.txt')
    with open(f'{config.data_dir}/train_candidates.txt', 'w', encoding='utf-8') as fp:
        fp.write('question_id,pos_ans_id,neg_ans_id\n')
        for item in train_candidates:
            fp.write(','.join(item)+'\n')

    # gen dev candidates
    dev_candidates = []
    with tqdm(total=len(dev_indexes)) as pbar:
        pbar.set_description('gen dev candidates')
        for index in dev_indexes:
            question_id = index
            ans_id = index
            cnt = 0
            label = 1
            dev_candidates.append([f'{question_id}', f'{ans_id}', f'{cnt}', f'{label}'])
            neg_index = get_neg_index(ans_id, max_num, pair_num=config.dev_pair_num-1)
            dev_candidates.extend(
                [
                    [f'{question_id}', f'{neg_index[i]}', f'{i+1}', '0']
                    for i in range(config.dev_pair_num-1)
                ]
            )
            pbar.update(1)
    print(f'save {config.data_dir}/dev_candidates.txt')
    with open(f'{config.data_dir}/dev_candidates.txt', 'w', encoding='utf-8') as fp:
        fp.write('question_id,ans_id,cnt,label\n')
        for item in dev_candidates:
            fp.write(','.join(item)+'\n')

    # gen test candidates
    test_candidates = []
    with tqdm(total=len(test_indexes)) as pbar:
        pbar.set_description('gen test candidates')
        for index in test_indexes:
            question_id = index
            ans_id = index
            cnt = 0
            label = 1
            test_candidates.append([f'{question_id}', f'{ans_id}', f'{cnt}', f'{label}'])
            neg_index = get_neg_index(ans_id, max_num, pair_num=config.test_pair_num - 1)
            test_candidates.extend(
                [
                    [f'{question_id}', f'{neg_index[i]}', f'{i + 1}', '0']
                    for i in range(config.test_pair_num - 1)
                ]
            )
            pbar.update(1)
    print(f'save {config.data_dir}/test_candidates.txt')
    with open(f'{config.data_dir}/test_candidates.txt', 'w', encoding='utf-8') as fp:
        fp.write('question_id,ans_id,cnt,label\n')
        for item in test_candidates:
            fp.write(','.join(item)+'\n')


class Config():
    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://10.23.11.253:27017")
        self.db = self.client['db_familyDoctor_QA_V2']
        self.data_dir = os.path.abspath('.') + '/data'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.new_db = self.client['cmcqa_epilepsy']

        self.seed = 1
        self.dev_rate = 0.1
        self.test_rate = 0.1
        self.train_pair_num = 113
        self.dev_pair_num = 113
        self.test_pair_num = 113
