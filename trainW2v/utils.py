import pickle
import sys
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