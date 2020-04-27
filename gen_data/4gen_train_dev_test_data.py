from data_pro_utils import *
from tqdm import tqdm
import os


def build_dict(data, file_class=None):
    data_dict = {}
    with tqdm(total=len(data)) as pbar:
        for index, item in enumerate(data):
            if file_class == 'question':
                data_dict[item.question_id] = {'content': item.content}
            elif file_class == 'answer':
                data_dict[item.ans_id] = {'question_id': item.question_id, 'content': item.content}
            pbar.update(1)
    return data_dict


def gen_train_text_data(question_dict, answer_dict, reload=True):
    if reload and os.path.exists(text_data_path['train_text_data']):
        train_text_data = load_pkl_data(text_data_path['train_text_data'])
        return train_text_data
    train_candidates = load_pkl_data(raw_data_path['train_candidates'])
    train_text_data = []
    with tqdm(total=len(train_candidates)) as pbar:
        pbar.set_description('building  train data')
        for index, item in enumerate(train_candidates):
            question_id = item.question_id
            pos_ans_id = item.pos_ans_id
            neg_ans_id = item.neg_ans_id

            q_content = question_dict[question_id]['content']
            p_ans_content = answer_dict[pos_ans_id]['content']
            p_ans_qid = answer_dict[pos_ans_id]['question_id']
            n_ans_content = answer_dict[neg_ans_id]['content']
            n_ans_pid = answer_dict[neg_ans_id]['question_id']

            if p_ans_qid != question_id or n_ans_pid == question_id:
                raise Exception('error in pid')
            else:
                train_text_data.append(
                    TrainTextExample(
                        question_id=question_id,
                        q_content=q_content,
                        pos_ans_id=pos_ans_id,
                        p_ans_content=p_ans_content,
                        neg_ans_id=neg_ans_id,
                        n_ans_content=n_ans_content,
                    )
                )
                pbar.update(1)
    save_pkl_data(train_text_data, text_data_path['train_text_data'])
    return train_text_data


def gen_dev_text_data(question_dict, answer_dict, reload=True):
    if reload and os.path.exists(text_data_path['dev_text_data']):
        dev_text_data = load_pkl_data(text_data_path['dev_text_data'])
        return dev_text_data
    dev_candidates = load_pkl_data(raw_data_path['dev_candidates'])
    dev_text_data = []
    with tqdm(total=len(dev_candidates)) as pbar:
        pbar.set_description('building  dev text data')
        for index, item in enumerate(dev_candidates):
            question_id = item.question_id
            ans_id = item.ans_id
            cnt = item.cnt
            label = item.label
            q_content = question_dict[question_id]['content']
            a_content = answer_dict[ans_id]['content']

            dev_text_data.append(
                DevTextExample(
                    question_id=question_id,
                    q_content=q_content,
                    ans_id=ans_id,
                    a_content=a_content,
                    cnt=cnt,
                    label=label,
                )
            )
            pbar.update(1)

    save_pkl_data(dev_text_data, text_data_path['dev_text_data'])
    return dev_text_data


def gen_test_text_data(question_dict, answer_dict, reload=True):
    if reload and os.path.exists(text_data_path['test_text_data']):
        test_text_data = load_pkl_data(text_data_path['test_text_data'])
        return test_text_data
    test_candidates = load_pkl_data(raw_data_path['test_candidates'])
    test_text_data = []
    with tqdm(total=len(test_candidates)) as pbar:
        pbar.set_description('building  test text data')
        for index, item in enumerate(test_candidates):
            question_id = item.question_id
            ans_id = item.ans_id
            cnt = item.cnt
            label = item.label
            q_content = question_dict[question_id]['content']
            a_content = answer_dict[ans_id]['content']

            test_text_data.append(
                TestTextExample(
                    question_id=question_id,
                    q_content=q_content,
                    ans_id=ans_id,
                    a_content=a_content,
                    cnt=cnt,
                    label=label,
                )
            )
            pbar.update(1)

    save_pkl_data(test_text_data, text_data_path['test_text_data'])
    return test_text_data


if __name__ == '__main__':
    root_path = os.path.abspath('.')
    raw_data_dir = f'{root_path}/raw_data'
    raw_data_path = {
        'train_candidates': f'{raw_data_dir}/train_candidates.pkl',
        'dev_candidates': f'{raw_data_dir}/dev_candidates.pkl',
        'test_candidates': f'{raw_data_dir}/test_candidates.pkl',
        'question': f'{raw_data_dir}/question.pkl',
        'answer': f'{raw_data_dir}/answer.pkl',
    }

    question = load_pkl_data(raw_data_path['question'])
    answer = load_pkl_data(raw_data_path['answer'])
    question_dict = build_dict(question, file_class='question')
    answer_dict = build_dict(answer, file_class='answer')

    data_dir = f'{root_path}/data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    text_data_path = {
        'train_text_data': f'{data_dir}/train_text_data.pkl',
        'dev_text_data': f'{data_dir}/dev_text_data.pkl',
        'test_text_data': f'{data_dir}/test_text_data.pkl',
    }

    gen_train_text_data(question_dict, answer_dict, reload=True)

    gen_dev_text_data(question_dict, answer_dict, reload=True)

    gen_test_text_data(question_dict, answer_dict, reload=True)
    print('end')
