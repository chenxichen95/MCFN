import pickle


def load_pkl_data(filePath):
    with open(filePath, 'rb') as fp:
        data_pkl = fp.read()
    return pickle.loads(data_pkl)


def save_pkl_data(data, filePath):
    data_pkl = pickle.dumps(data)
    with open(filePath, 'wb') as fp:
        fp.write(data_pkl)


def get_lc(filepath):
    count = -1
    with open(filepath, 'r', encoding='utf-8') as fp:
        for line in fp:
            count += 1
    return count


class RawExample(object):
    def __init__(self, ans_id=None, pos_ans_id=None, neg_ans_id=None, cnt=None, question_id=None, content=None, label=None):

        self.question_id = question_id
        self.ans_id = ans_id
        self.pos_ans_id = pos_ans_id
        self.neg_ans_id = neg_ans_id
        self.content = content
        self.cnt = cnt
        self.label = label

    def __str__(self):
        self.__repr__()

    def __repr__(self):
        s = "{"
        if self.question_id:
            s += f'question_id: {self.question_id}'
        if self.ans_id:
            s += f'||ans_id: {self.ans_id}'
        if self.pos_ans_id:
            s += f'||pos_ans_id: {self.pos_ans_id}'
        if self.neg_ans_id:
            s += f'||neg_ans_id: {self.neg_ans_id}'
        if self.content:
            s += f'||content: {self.content}'
        if self.cnt:
            s += f'||cnt: {self.cnt}'
        if self.label:
            s += f'||label: {self.label}'
        s += "}"

        return s


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

