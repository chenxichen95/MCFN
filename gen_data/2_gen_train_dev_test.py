from utils import *

if __name__ == '__main__':
    config = Config()
    gen_question_answer_csv(config.new_db, config.data_dir, max_num=133)
    gen_candidates2(max_num=113, config=config)