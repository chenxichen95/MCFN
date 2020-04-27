import config
import os
from utils import *
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig
from pytorch_transformers import AdamW, WarmupLinearSchedule
import time
import logging
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from model_utils import load_best_model, save_model
from OnlyBiGRU import OnlyBiGRU
from MCFN import MCFN
from MSCNN import MSCNN
from MSAIN import MSAIN
from MLCCNN import MLCCNN

import operator as op
import gc


class Log():
    def __init__(self, args, running_time, baseline, benchmark):
        self.log_dir_name = running_time
        self.save_dir = os.path.join(args.output_dir, f'{baseline}', self.log_dir_name, f'benchmark-{benchmark}')
        if os.path.exists(self.save_dir) == False:
            os.makedirs(self.save_dir, exist_ok=True)

    def print(self, str):
        with open(self.save_dir+'/train-dev.log', 'a+', encoding='utf-8') as fp:
            fp.write(str+'\n')



def train(train_features_tensor_dataset, dev_features_tensor_dataset, test_features_tensor_dataset=None, data_mode='mongo', demo=False, use_w2v=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.batch_size = int(args.batch_size)

    if 'set random seed':
        seed_everything(args.seed)
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("train examples {}".format(len(train_features_tensor_dataset)))
    num_train_steps = int(len(train_features_tensor_dataset) / args.batch_size * args.num_train_epochs)
    steps_a_epoch = int(num_train_steps // args.num_train_epochs)

    if 'create train TensorDataset':
        train_sampler = RandomSampler(train_features_tensor_dataset)
        train_dataloader = DataLoader(
            train_features_tensor_dataset,
            sampler=train_sampler,
            batch_size=args.batch_size,
            drop_last=True
        )

    if 'create dev TensorDataset':
        dev_sampler = SequentialSampler(dev_features_tensor_dataset)
        dev_dataloader = DataLoader(
            dev_features_tensor_dataset,
            sampler=dev_sampler,
            batch_size=args.batch_size,
            drop_last=False
        )

    if 'create test TensorDataset':
        test_sampler = SequentialSampler(test_features_tensor_dataset)
        test_dataloader = DataLoader(
            test_features_tensor_dataset,
            sampler=test_sampler,
            batch_size=args.batch_size,
            drop_last=False
        )

    log = Log(args, running_time, baseline, benchmark)
    if 'create model':
        bert_config = BertConfig.from_json_file(args.bert_config_file)
        weight = ''
        if use_w2v:
            weight = load_pkl_data(args.w2v_weight)
        if baseline == 'baseline1':
            model = Baseline1(args, use_w2v=use_w2v, weight=weight)
        elif baseline == 'baseline2':
            model = Baseline2(args, use_w2v=use_w2v, weight=weight)
        elif baseline == 'baseline3':
            model = Baseline3(args, use_w2v=use_w2v, weight=weight)
        elif baseline == 'baseline4':
            model = Baseline4(args, use_w2v=use_w2v, weight=weight)
        elif baseline == 'baseline5':
            model = Baseline5(args, use_w2v=use_w2v, weight=weight)
        elif baseline == 'baseline6':
            model = Baseline6(args, use_w2v=use_w2v, weight=weight)
        elif baseline == 'MSCNN':
            model = MSCNN(args, use_w2v=use_w2v, weight=weight)
        elif baseline == 'MSAIN':
            model = MSAIN(args, use_w2v=use_w2v, weight=weight)
        elif baseline == 'MLCCNN':
            model = MLCCNN(args, use_w2v=use_w2v, weight=weight)
        else:
            raise Exception('model not exsit')

        model.to(device)

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer], 'weight_decay': 0.0}]
        optimizer = torch.optim.Adagrad(optimizer_grouped_parameters, lr=args.learning_rate)

        # DataParallel training
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

    global_step = 0
    best_top1 = 0
    for curEpoch in range(int(args.num_train_epochs)):
        model.train()
        with tqdm(total=steps_a_epoch) as pbar:
            for step, batch in enumerate(train_dataloader):
                if data_mode == 'mongo':
                    indexes = batch[0].sort().values.tolist()
                    batch_data = [item for item in collection.find({'index': {'$in': indexes}})]
                    if use_w2v:
                        q_input_ids_w2v = torch.tensor(
                        [item['q_w2v_ids'] for item in batch_data],
                        dtype=torch.long).to(device)
                        p_a_input_ids_w2v = torch.tensor(
                            [item['p_a_w2v_ids'] for item in batch_data],
                            dtype=torch.long).to(device)
                        n_a_input_ids_w2v = torch.tensor(
                            [item['n_a_w2v_ids'] for item in batch_data],
                            dtype=torch.long).to(device)
                        q_input_mask_tensor = torch.tensor(
                            [item['q_input_mask'] for item in batch_data],
                            dtype=torch.uint8
                        ).to(device)
                        p_a_input_mask_tensor = torch.tensor(
                            [item['p_a_input_mask'] for item in batch_data],
                            dtype=torch.uint8
                        ).to(device)
                        n_a_input_mask_tensor = torch.tensor(
                            [item['n_a_input_mask'] for item in batch_data],
                            dtype=torch.uint8
                        ).to(device)
                    else:
                        q_input_ids_tensor = torch.tensor(
                            [item['q_input_ids'] for item in batch_data],
                            dtype=torch.long
                        ).to(device)
                        q_input_mask_tensor = torch.tensor(
                            [item['q_input_mask'] for item in batch_data],
                            dtype=torch.long
                        ).to(device)
                        q_segment_ids_tensor = torch.tensor(
                            [item['q_segment_ids'] for item in batch_data],
                            dtype=torch.long
                        ).to(device)
                        p_a_input_ids_tensor = torch.tensor(
                            [item['p_a_input_ids'] for item in batch_data],
                            dtype=torch.long
                        ).to(device)
                        p_a_input_mask_tensor = torch.tensor(
                            [item['p_a_input_mask'] for item in batch_data],
                            dtype=torch.long
                        ).to(device)
                        p_a_segment_ids_tensor = torch.tensor(
                            [item['p_a_segment_ids'] for item in batch_data],
                            dtype=torch.long
                        ).to(device)
                        n_a_input_ids_tensor = torch.tensor(
                            [item['n_a_input_ids'] for item in batch_data],
                            dtype=torch.long
                        ).to(device)
                        n_a_input_mask_tensor = torch.tensor(
                            [item['n_a_input_mask'] for item in batch_data],
                            dtype=torch.long
                        ).to(device)
                        n_a_segment_ids_tensor = torch.tensor(
                            [item['n_a_segment_ids'] for item in batch_data],
                            dtype=torch.long
                        ).to(device)
                elif data_mode == 'local':
                    batch = tuple(input.to(device) for input in batch)
                    question_id, pos_ans_id, neg_ans_id, \
                    q_input_ids, q_input_mask, q_segment_ids, \
                    p_a_input_ids, p_a_input_mask, p_a_segment_ids, \
                    n_a_input_ids, n_a_input_mask, n_a_segment_ids = batch

                if use_w2v:
                    loss, pre_pos_sim = model(
                        q_input_ids_w2v=q_input_ids_w2v,
                        p_a_input_ids_w2v=p_a_input_ids_w2v,
                        n_a_input_ids_w2v=n_a_input_ids_w2v,
                        q_input_mask=q_input_mask_tensor,
                        p_a_input_mask=p_a_input_mask_tensor,
                        n_a_input_mask=n_a_input_mask_tensor,
                    )
                else:
                    loss, pre_pos_sim = model(
                        q_input_ids=q_input_ids_tensor,
                        q_input_mask=q_input_mask_tensor,
                        q_segment_ids=q_segment_ids_tensor,
                        p_a_input_ids=p_a_input_ids_tensor,
                        p_a_input_mask=p_a_input_mask_tensor,
                        p_a_segment_ids=p_a_segment_ids_tensor,
                        n_a_input_ids=n_a_input_ids_tensor,
                        n_a_input_mask=n_a_input_mask_tensor,
                        n_a_segment_ids=n_a_segment_ids_tensor
                    )
                if args.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()
                if 'clip grad norm':
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                #lr_scheduler.step()   # initial_lr = 0
                optimizer.step()
                optimizer.zero_grad()

                pbar.set_description(f'{baseline}, benchmark-{benchmark}, train loss: {loss:.6f}')
                log.print(f'{baseline}, benchmark-{benchmark}, step: {global_step}, train loss: {loss:.6f}')
                pbar.update(1)
                global_step += 1

                if step % args.dev_step == 0 and step != 0:
                    top1 = dev(args, model, dev_dataloader, device, demo=False, use_w2v=use_w2v)
                    if top1 > best_top1:
                        best_top1 = top1
                        logging.info(
                            f'{baseline}, benchmark-{benchmark}, epoch:{curEpoch}, global_step:{global_step}, dev top1:{top1:.2%}, best top1: {best_top1:.2%}')
                        log.print(
                            f'{baseline}, benchmark-{benchmark}, epoch:{curEpoch}, global_step:{global_step}, dev top1:{top1:.2%}, best top1: {best_top1:.2%}')

                        save_model(model, global_step, args, logging, log, running_time, benchmark, baseline)
                    else:
                        logging.info(
                            f'{baseline}, benchmark-{benchmark}, epoch:{curEpoch}, global_step:{global_step}, dev top1:{top1:.2%}, best top1: {best_top1:.2%}')
                        log.print(f'{baseline}, benchmark-{benchmark}, epoch:{curEpoch}, global_step:{global_step}, dev top1:{top1:.2%}, best top1: {best_top1:.2%}')

                    model.train()

        top1 = dev(args, model, dev_dataloader, device, demo=False, use_w2v=use_w2v)
        if top1 > best_top1:
            best_top1 = top1
            save_model(model, global_step, args, logging, log, running_time, benchmark, baseline)
        logging.info(f'{baseline}, benchmark-{benchmark}, epoch:{curEpoch}, global_step:{global_step}, dev top1:{top1:.2%}, best top1: {best_top1:.2%}')
        log.print(
            f'{baseline}, benchmark-{benchmark}, epoch:{curEpoch}, global_step:{global_step}, dev top1:{top1:.2%}, best top1: {best_top1:.2%}')

    test_top1 = test(args, model, test_dataloader, device, running_time, baseline, benchmark, demo=False,
                     use_w2v=use_w2v)
    logging.info(
        f'{baseline}, benchmark-{benchmark}, best test top1:{test_top1:.2%}')
    log.print(
        f'{baseline}, benchmark-{benchmark}, best test top1:{test_top1:.2%}')


def test(args, model, test_dataloader, device, running_time, baseline, benchmark, demo=False, use_w2v=False):
    torch.cuda.empty_cache()
    model = load_best_model(model, args, running_time, baseline, benchmark)
    model.eval()
    question_id_set = []
    label_id_set = []
    pre_pos_score_set = []
    with tqdm(total=int(len(dev_features_tensor_dataset) / args.batch_size)) as pbar:
        pbar.set_description('test')
        for step, batch in enumerate(test_dataloader):
            batch = (t.to(device) for t in batch)
            if use_w2v:
                question_id_tensor, ans_id_tensor, \
                q_input_ids_tensor, q_input_mask_tensor, q_segment_ids_tensor, \
                a_input_ids_tensor, a_input_mask_tensor, a_segment_ids_tensor, \
                cnt_tensor, label_tensor, \
                q_w2v_ids, a_w2v_ids = batch
                pre_pos_score = model(
                    q_input_ids_w2v=q_w2v_ids,
                    p_a_input_ids_w2v=a_w2v_ids,
                    q_input_mask=q_input_mask_tensor.byte(),
                    p_a_input_mask=a_input_mask_tensor.byte(),
                )
            else:
                question_id_tensor, ans_id_tensor, \
                q_input_ids_tensor, q_input_mask_tensor, q_segment_ids_tensor, \
                a_input_ids_tensor, a_input_mask_tensor, a_segment_ids_tensor, \
                cnt_tensor, label_tensor = batch
                pre_pos_score = model(
                    q_input_ids=q_input_ids_tensor,
                    q_input_mask=q_input_mask_tensor,
                    q_segment_ids=q_segment_ids_tensor,
                    p_a_input_ids=a_input_ids_tensor,
                    p_a_input_mask=a_input_mask_tensor,
                    p_a_segment_ids=a_segment_ids_tensor,
                )
            question_id_set.extend(question_id_tensor.detach().tolist())
            label_id_set.extend(label_tensor.detach().tolist())
            pre_pos_score_set.extend(pre_pos_score.detach().tolist())
            if demo:
                if step == 50:
                    break
            pbar.update(1)
    results = {}
    for i in range(len(question_id_set)):
        if results.get(str(question_id_set[i]), 'none') == 'none':
            results[str(question_id_set[i])] = [[label_id_set[i], pre_pos_score_set[i]]]
        else:
            results[str(question_id_set[i])].append([label_id_set[i], pre_pos_score_set[i]])
    total_num = 0
    corr_num = 0
    for key, item in results.items():
        total_num += 1
        item.sort(key=op.itemgetter(1), reverse=True)
        if item[0][0] == 1:
            corr_num += 1
    top1 = corr_num * 1.0 / total_num

    return top1


def dev(args, model, dev_dataloader, device, demo=False, use_w2v=False):
    torch.cuda.empty_cache()
    model.eval()
    question_id_set = []
    label_id_set = []
    pre_pos_score_set = []
    with tqdm(total=int(len(dev_features_tensor_dataset)/args.batch_size)) as pbar:
        pbar.set_description('dev')
        for step, batch in enumerate(dev_dataloader):
            batch = (t.to(device) for t in batch)
            if use_w2v:
                question_id_tensor, ans_id_tensor, \
                q_input_ids_tensor, q_input_mask_tensor, q_segment_ids_tensor, \
                a_input_ids_tensor, a_input_mask_tensor, a_segment_ids_tensor, \
                cnt_tensor, label_tensor, \
                q_w2v_ids, a_w2v_ids = batch
                pre_pos_score = model(
                    q_input_ids_w2v=q_w2v_ids,
                    p_a_input_ids_w2v=a_w2v_ids,
                    q_input_mask=q_input_mask_tensor.byte(),
                    p_a_input_mask=a_input_mask_tensor.byte(),
                )
            else:
                question_id_tensor, ans_id_tensor, \
                q_input_ids_tensor, q_input_mask_tensor, q_segment_ids_tensor, \
                a_input_ids_tensor, a_input_mask_tensor, a_segment_ids_tensor, \
                cnt_tensor, label_tensor = batch
                pre_pos_score = model(
                    q_input_ids=q_input_ids_tensor,
                    q_input_mask=q_input_mask_tensor,
                    q_segment_ids=q_segment_ids_tensor,
                    p_a_input_ids=a_input_ids_tensor,
                    p_a_input_mask=a_input_mask_tensor,
                    p_a_segment_ids=a_segment_ids_tensor,
                )
            question_id_set.extend(question_id_tensor.detach().tolist())
            label_id_set.extend(label_tensor.detach().tolist())
            pre_pos_score_set.extend(pre_pos_score.detach().tolist())
            if demo:
                if step == 50:
                    break
            pbar.update(1)
    results = {}
    for i in range(len(question_id_set)):
        if results.get(str(question_id_set[i]), 'none') == 'none':
            results[str(question_id_set[i])] = [[label_id_set[i], pre_pos_score_set[i]]]
        else:
            results[str(question_id_set[i])].append([label_id_set[i], pre_pos_score_set[i]])
    total_num = 0
    corr_num = 0
    for key, item in results.items():
        total_num += 1
        item.sort(key=op.itemgetter(1), reverse=True)
        if item[0][0] == 1:
            corr_num += 1
    top1 = corr_num * 1.0 / total_num

    return top1


if __name__ == "__main__":
    args = config.Config().args
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client['cmcqa_epilepsy']

    use_w2v = True
    if use_w2v:
        collection = db['train_use_w2v']
    else:
        collection = db['train']

    #train_features_tensor_dataset = get_tensor_dataset(args, file_class='train')
    train_sample_index_dataset = get_tensor_dataset(args, file_class='train', save_mode='mongo', use_w2v=use_w2v)
    dev_features_tensor_dataset = get_tensor_dataset(args, file_class='dev', use_w2v=use_w2v)
    test_features_tensor_dataset = get_tensor_dataset(args, file_class='test', use_w2v=use_w2v)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    running_time = time.strftime("%Y-%m-%d@%H_%M_%S", time.localtime())

    #baselines = ['baseline1', 'baseline2', 'baseline3', 'baseline4', 'baseline5', 'baseline6']
    baselines = ['MCFN']
    for baseline in baselines:
        for benchmark in range(args.benchmark):
            args.seed = 40 + benchmark
            train(train_sample_index_dataset, dev_features_tensor_dataset, test_features_tensor_dataset=test_features_tensor_dataset, demo=False, use_w2v=use_w2v)
            torch.cuda.empty_cache()
            gc.collect()
