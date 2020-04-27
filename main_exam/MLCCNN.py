import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import os
import math


class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(x * 0.7978845608 * (1. + 0.044715 * x * x)))

class SingleScaleCNN(nn.Module):
    def __init__(self, args):
        super(SingleScaleCNN, self).__init__()
        self.cnn_layer1 = nn.Conv1d(
            in_channels=300,
            out_channels=args.cnn_kernel_num,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=True,
            padding_mode='zero',
        )
        nn.init.normal_(self.cnn_layer1.weight, std=math.sqrt(2. / (300 * 1)))
        nn.init.zeros_(self.cnn_layer1.bias)
        self.cnn_layer1 = nn.Sequential(nn.utils.weight_norm(self.cnn_layer1), GeLU())

    def forward(self, input_tensor, mask):
        input_tensor = input_tensor.permute([0, 2, 1])
        mask = mask.permute([0, 2, 1])
        input_tensor = input_tensor.masked_fill_(~mask, .0)
        input_tensor1 = self.cnn_layer1(input_tensor)
        return input_tensor1.permute([0, 2, 1])

class CompositeCNN(nn.Module):
    def __init__(self, args):
        super(CompositeCNN, self).__init__()
        in_channels = 300

        self.cnn_layer1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=args.cnn_kernel_num,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=True,
            padding_mode='zero',
        )
        nn.init.normal_(self.cnn_layer1.weight, std=math.sqrt(2. / (in_channels * 1)))
        nn.init.zeros_(self.cnn_layer1.bias)
        self.cnn_layer1 = nn.Sequential(nn.utils.weight_norm(self.cnn_layer1), GeLU())

        self.cnn_layer2 = nn.Conv1d(
            in_channels=args.cnn_kernel_num,
            out_channels=args.cnn_kernel_num,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=True,
            padding_mode='zero',
        )
        nn.init.normal_(self.cnn_layer2.weight, std=math.sqrt(2. / (in_channels * 2)))
        nn.init.zeros_(self.cnn_layer2.bias)
        self.cnn_layer2 = nn.Sequential(nn.utils.weight_norm(self.cnn_layer2), GeLU())

        self.cnn_layer3 = nn.Conv1d(
            in_channels=args.cnn_kernel_num,
            out_channels=args.cnn_kernel_num,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=True,
            padding_mode='zero',
        )
        nn.init.normal_(self.cnn_layer3.weight, std=math.sqrt(2. / (in_channels * 3)))
        nn.init.zeros_(self.cnn_layer3.bias)
        self.cnn_layer3 = nn.Sequential(nn.utils.weight_norm(self.cnn_layer3), GeLU())

    def forward(self, input_tensor, mask):
        input_tensor = input_tensor.permute([0, 2, 1])
        mask = mask.permute([0, 2, 1])
        input_tensor = input_tensor.masked_fill_(~mask, .0)

        input_tensor1 = self.cnn_layer1(input_tensor)
        input_tensor2 = self.cnn_layer2(input_tensor1)
        input_tensor3 = self.cnn_layer3(input_tensor2)
        return input_tensor1.permute([0, 2, 1]), input_tensor2.permute([0, 2, 1]), input_tensor3.permute([0, 2, 1])



class MultiScaleCNN(nn.Module):
    def __init__(self, args):
        super(MultiScaleCNN, self).__init__()
        in_channels = 300

        self.cnn_layer1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=args.cnn_kernel_num,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=True,
            padding_mode='zero',
        )
        nn.init.normal_(self.cnn_layer1.weight, std=math.sqrt(2. / (in_channels * 1)))
        nn.init.zeros_(self.cnn_layer1.bias)
        self.cnn_layer1 = nn.Sequential(nn.utils.weight_norm(self.cnn_layer1), GeLU())

        self.cnn_layer2 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=args.cnn_kernel_num,
            kernel_size=2,
            stride=1,
            padding=0,
            groups=1,
            bias=True,
            padding_mode='zero',
        )
        nn.init.normal_(self.cnn_layer2.weight, std=math.sqrt(2. / (in_channels * 2)))
        nn.init.zeros_(self.cnn_layer2.bias)
        self.cnn_layer2 = nn.Sequential(nn.utils.weight_norm(self.cnn_layer2), GeLU())

        self.cnn_layer3 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=args.cnn_kernel_num,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=True,
            padding_mode='zero',
        )
        nn.init.normal_(self.cnn_layer3.weight, std=math.sqrt(2. / (in_channels * 3)))
        nn.init.zeros_(self.cnn_layer3.bias)
        self.cnn_layer3 = nn.Sequential(nn.utils.weight_norm(self.cnn_layer3), GeLU())

    def forward(self, input_tensor, mask):
        input_tensor = input_tensor.permute([0, 2, 1])
        mask = mask.permute([0, 2, 1])
        input_tensor = input_tensor.masked_fill_(~mask, .0)
        input_tensor1 = self.cnn_layer1(input_tensor)
        input_tensor2 = self.cnn_layer2(torch.cat([input_tensor, torch.zeros(input_tensor.size(0), input_tensor.size(1), 1, device=input_tensor.device)], dim=-1))
        input_tensor3 = self.cnn_layer3(input_tensor)
        return input_tensor1.permute([0, 2, 1]), input_tensor2.permute([0, 2, 1]), input_tensor3.permute([0, 2, 1])


class OneScaleAttentionLayer(nn.Module):
    def __init__(self, args):
        super(OneScaleAttentionLayer, self).__init__()
        #self.U = nn.init.normal(torch.Tensor(args.cnn_kernel_num, args.cnn_kernel_num), mean=0, std=0.02)
        #self.U.requires_grad = True
        self.hidden_size = args.cnn_kernel_num
        self.U = nn.parameter.Parameter(
            nn.init.normal(torch.Tensor(self.hidden_size, self.hidden_size), mean=0, std=0.02),
            requires_grad=True,
        )

    def forward(self, q, a, q_mask, a_mask):
        # q.shape = (N, Tq, args.cnn_kernel_num)
        # a.shape = (N, Ta, args.cnn_kernel_num)
        # q_mask.shape = (N, Tq, 1)
        # a_mask.shape = (N, Ta, 1)
        mask = torch.matmul(q_mask.float(), a_mask.permute([0, 2, 1]).float()).byte()
        X = torch.matmul(torch.matmul(q, self.U), a.permute([0, 2, 1]))
        X.masked_fill_(~mask, -1e7)
        I = torch.sigmoid(X) # (N, Tq, Ta)

        # a to q attention
        att_a2q = F.softmax(I, dim=-1).max(dim=-1)[0]  # (N, Tq)
        # q to a attention
        att_q2a = F.softmax(I, dim=-2).max(dim=-2)[0]  # (N, Ta)

        # attentive pooling
        q_attn = torch.einsum('abc,ab->ac', q, att_a2q)
        a_attn = torch.einsum('abc,ab->ac', a, att_q2a)

        return q_attn, a_attn

    def extra_repr(self):
        return f'hidden_size=({self.hidden_size}, {self.hidden_size})'


class MarginLoss(nn.Module):
    def __init__(self, args):
        super(MarginLoss, self).__init__()
        self.margin = args.margin
        self.eps = args.eps
        self.reduction = args.reduction


    def forward(self, q_pos, a_pos, q_neg=None, a_neg=None):
        '''
            q_pos.shape = (N, H)
            a_pos.shape = (N, H)
            q_neg.shape = (N, H)
            a_neg.shape = (N, H)
        '''
        if isinstance(q_neg, torch.Tensor):  # train
            qa_pos_score = torch.abs(F.cosine_similarity(q_pos, a_pos, dim=-1, eps=self.eps))   # shape = (N)
            qa_neg_score = torch.abs(F.cosine_similarity(q_neg, a_neg, dim=-1, eps=self.eps))   # shape = (N)

            score = self.margin - qa_pos_score + qa_neg_score
            score = torch.where(score < 0, torch.full_like(score, 0), score)

            loss = 0
            if self.reduction == 'sum':
                loss = score.sum()
            elif self.reduction == 'mean':
                loss = score.mean()
            elif self.reduction == 'none':
                return score, qa_pos_score

            return loss, qa_pos_score
        else:  # eval
            qa_pos_score = torch.abs(F.cosine_similarity(q_pos, a_pos, dim=-1, eps=self.eps))   # shape = (N)
            return qa_pos_score

    def extra_repr(self):
        return f'margn={self.margin}, eps={self.eps}'


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional):
        super(BiGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.bigru = torch.nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,  # 一定要置 0 ，不使用这个类封装的 dropout ，否则会导致随机种子失效
            bidirectional=self.bidirectional
        )

    def forward(self, x, lengths):
        ori_len = x.shape[1]
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)  # 打包
        out, h_n = self.bigru(x)
        out = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=ori_len)  # 拆包
        return out[0]


class Fusion_Layer(nn.Module):
    def __init__(self, args):
        super(Fusion_Layer, self).__init__()
        self.hidden_size = args.gru_hidden_size
        self.G1 = Linear(args.cnn_kernel_num*2, args.gru_hidden_size, activations=True)
        self.G2 = Linear(args.cnn_kernel_num*2, args.gru_hidden_size, activations=True)
        self.G3 = Linear(args.cnn_kernel_num*2, args.gru_hidden_size, activations=True)
        self.G = Linear(args.gru_hidden_size*3, args.gru_hidden_size, activations=True)
        self.pooling = Pooling()

    def forward(self, input, mask):
        a1 = self.G1(torch.cat([input[0], input[1]], dim=-1))
        a2 = self.G2(torch.cat([input[0], input[0] - input[1]], dim=-1))
        a3 = self.G3(torch.cat([input[0], input[0] * input[1]], dim=-1))
        a = self.G(torch.cat([a1, a2, a3], dim=-1))
        a_pool = self.pooling(a, mask)
        return a_pool


class Pooling(nn.Module):
    def forward(self, inputs, mask):
        return inputs.max(dim=1)[0]


class OneScaleOneAttentionLayer(nn.Module):
    def __init__(self, args):
        super(OneScaleOneAttentionLayer, self).__init__()
        self.hidden_size = args.cnn_kernel_num

        # Semantic attention
        self.temperature = nn.Parameter(torch.tensor(1 / math.sqrt(self.hidden_size)))

    def _attention(self, q, a):
        return torch.matmul(q, a.transpose(1, 2)) * self.temperature

    def forward(self, q, a, q_mask, a_mask):
        # q.shape = (N, Tq, args.cnn_kernel_num)
        # a.shape = (N, Ta, args.cnn_kernel_num)

        attn = self._attention(q, a)  # (N, Tq, Ta)
        mask = torch.matmul(q_mask.float(), a_mask.transpose(1, 2).float()).byte()
        attn.masked_fill_(~mask, -1e7)
        attn_q = F.softmax(attn, dim=1)
        attn_a = F.softmax(attn, dim=2)
        a_s = torch.matmul(attn_q.transpose(1, 2), q)
        q_s = torch.matmul(attn_a, a)

        return torch.cat([q, q_s], dim=-1), torch.cat([a, a_s], dim=-1)


class OneScaleOneAttentionLayer2(nn.Module):
    def __init__(self, args):
        super(OneScaleOneAttentionLayer2, self).__init__()
        #self.U = nn.init.normal(torch.Tensor(args.cnn_kernel_num, args.cnn_kernel_num), mean=0, std=0.02)
        #self.U.requires_grad = True
        self.hidden_size = args.cnn_kernel_num

        # Semantic attention
        self.temperature = nn.Parameter(torch.tensor(1 / math.sqrt(self.hidden_size)))

    def _attention(self, a, b):
        return torch.matmul(a, b.transpose(1, 2)) * self.temperature


    def forward(self, q, a, q_mask, a_mask):
        # q.shape = (N, Tq, args.cnn_kernel_num)
        # a.shape = (N, Ta, args.cnn_kernel_num)
        # q_mask.shape = (N, Tq, 1)
        # a_mask.shape = (N, Ta, 1)

        attn = self._attention(q, a)
        mask = torch.matmul(q_mask.float(), a_mask.transpose(1, 2).float()).byte()
        attn.masked_fill_(~mask, -1e7)
        attn_q = F.softmax(attn, dim=1)
        attn_a = F.softmax(attn, dim=2)
        feature_a = torch.matmul(attn_q.transpose(1, 2), q)
        feature_q = torch.matmul(attn_a, a)

        return ([q, feature_q], [a, feature_a])


    def extra_repr(self):
        return f'hidden_size=({self.hidden_size}, {self.hidden_size})'


class Linear(nn.Module):
    def __init__(self, in_features, out_features, activations=False):
        super().__init__()
        linear = nn.Linear(in_features, out_features)
        nn.init.normal_(linear.weight, std=math.sqrt((2. if activations else 1.) / in_features))
        nn.init.zeros_(linear.bias)
        modules = [nn.utils.weight_norm(linear)]
        if activations:
            modules.append(GeLU())
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class ScoreLayer(nn.Module):
    def __init__(self, args):
        super(ScoreLayer, self).__init__()
        self.score_hs = args.score_hs
        self.output_layer = nn.Linear(args.cnn_kernel_num*3*2, 1, bias=False)
            #Linear(args.cnn_kernel_num*3*3, 1)

    def forward(self, inputs):
        # inputs.shape = (N, T, H)
        # outputs.shape = (N, T, 1)
        score = self.output_layer(inputs)
        #score = F.sigmoid(score)

        return score.squeeze(dim=-1)


class MarginLoss_kown_score(nn.Module):
    def __init__(self, args):
        super(MarginLoss_kown_score, self).__init__()
        self.margin = args.margin
        self.eps = args.eps
        self.reduction = args.reduction


    def forward(self, qa_pos_score, qa_neg_score=None):
        '''
            q_pos.shape = (N, H)
            a_pos.shape = (N, H)
            q_neg.shape = (N, H)
            a_neg.shape = (N, H)
        '''
        if isinstance(qa_neg_score, torch.Tensor):  # train
            qa_pos_score = qa_pos_score * 0.5 + 0.5   # shape = (N)
            qa_neg_score = qa_neg_score * 0.5 + 0.5   # shape = (N)

            score = self.margin - qa_pos_score + qa_neg_score
            score = torch.where(score < 0, torch.full_like(score, 0), score)

            loss = 0
            if self.reduction == 'sum':
                loss = score.sum()
            elif self.reduction == 'mean':
                loss = score.mean()
            elif self.reduction == 'none':
                return score, qa_pos_score

            return loss, qa_pos_score
        else:  # eval
            qa_pos_score = qa_pos_score * 0.5 + 0.5   # shape = (N)
            return qa_pos_score


    def extra_repr(self):
        return f'margn={self.margin}, eps={self.eps}'


class MLCCNN(nn.Module):
    def __init__(self, args, use_w2v=False, weight=None):
        super(MLCCNN, self).__init__()
        self.use_w2v = use_w2v
        if self.use_w2v:
            weight = np.array(weight, dtype=np.float32)
            self.w2v_embedding = torch.nn.Embedding(weight.shape[0], weight.shape[1], padding_idx=0, _weight=torch.from_numpy(weight))
            self.w2v_embedding.weight.requires_grad = False
        else:
            pass
        self.cnn = CompositeCNN(args)

        self.pooling = Pooling()
        self.loss = MarginLoss(args)

    def forward(
            self,
            q_input_ids=None,
            q_input_mask=None,
            q_segment_ids=None,
            p_a_input_ids=None,
            p_a_input_mask=None,
            p_a_segment_ids=None,
            n_a_input_ids=None,
            n_a_input_mask=None,
            n_a_segment_ids=None,
            q_input_ids_w2v=None,
            p_a_input_ids_w2v=None,
            n_a_input_ids_w2v=None,
    ):

        q_cur_maxlen = q_input_mask.sum(dim=-1).max()
        q_input_ids_w2v = q_input_ids_w2v[:, 0:q_cur_maxlen]
        q_input_mask = q_input_mask[:, 0:q_cur_maxlen]

        p_a_cur_maxlen = p_a_input_mask.sum(dim=-1).max()
        p_a_input_ids_w2v = p_a_input_ids_w2v[:, 0:p_a_cur_maxlen]
        p_a_input_mask = p_a_input_mask[:, 0:p_a_cur_maxlen]

        if isinstance(n_a_input_ids_w2v, torch.Tensor):
            n_a_cur_maxlen = n_a_input_mask.sum(dim=-1).max()
            n_a_input_ids_w2v = n_a_input_ids_w2v[:, 0:n_a_cur_maxlen]
            n_a_input_mask = n_a_input_mask[:, 0:n_a_cur_maxlen]

        q_embedding = self.w2v_embedding(q_input_ids_w2v)
        p_a_embedding = self.w2v_embedding(p_a_input_ids_w2v)
        if isinstance(n_a_input_ids_w2v, torch.Tensor):
            n_a_embedding = self.w2v_embedding(n_a_input_ids_w2v)


        q_input_mask = q_input_mask.unsqueeze(dim=-1)
        p_a_input_mask = p_a_input_mask.unsqueeze(dim=-1)
        if isinstance(n_a_input_ids, torch.Tensor) or isinstance(n_a_input_ids_w2v, torch.Tensor):
            n_a_input_mask = n_a_input_mask.unsqueeze(dim=-1)

        # multi-scale cnn
        q_cnn1, q_cnn2, q_cnn3 = self.cnn(q_embedding, q_input_mask)
        p_a_cnn1, p_a_cnn2, p_a_cnn3 = self.cnn(p_a_embedding, p_a_input_mask)
        if isinstance(n_a_input_ids, torch.Tensor) or isinstance(n_a_input_ids_w2v, torch.Tensor):
            n_a_cnn1, n_a_cnn2, n_a_cnn3 = self.cnn(n_a_embedding, n_a_input_mask)


        # max pooling
        q_pooling_1 = self.pooling(q_cnn1, q_input_mask)
        q_pooling_2 = self.pooling(q_cnn2, q_input_mask)
        q_pooling_3 = self.pooling(q_cnn3, q_input_mask)
        q_pooling = torch.cat([q_pooling_1, q_pooling_2, q_pooling_3], dim=-1)

        p_a_pooling_1 = self.pooling(p_a_cnn1, p_a_input_mask)
        p_a_pooling_2 = self.pooling(p_a_cnn2, p_a_input_mask)
        p_a_pooling_3 = self.pooling(p_a_cnn3, p_a_input_mask)
        p_a_pooling = torch.cat([p_a_pooling_1, p_a_pooling_2, p_a_pooling_3], dim=-1)
        if isinstance(n_a_input_ids, torch.Tensor) or isinstance(n_a_input_ids_w2v, torch.Tensor):
            n_a_pooling_1 = self.pooling(n_a_cnn1, n_a_input_mask)
            n_a_pooling_2 = self.pooling(n_a_cnn2, n_a_input_mask)
            n_a_pooling_3 = self.pooling(n_a_cnn3, n_a_input_mask)
            n_a_pooling = torch.cat([n_a_pooling_1, n_a_pooling_2, n_a_pooling_3], dim=-1)

        # calc loss
        if isinstance(n_a_input_ids, torch.Tensor) or isinstance(n_a_input_ids_w2v, torch.Tensor):
            loss, score = self.loss(q_pooling, p_a_pooling, q_pooling, n_a_pooling)
            return loss, score
        else:
            score = self.loss(q_pooling, p_a_pooling)
            return score


def update_lr(opt, args, step):
    ratio = 1
    if args.lr_decay_rate < 1.:
        args = args
        t = step
        base_ratio = args.min_lr / args.learning_rate
        if t < args.lr_warmup_steps:
            ratio = base_ratio + (1. - base_ratio) / max(1., args.lr_warmup_steps) * t
        else:
            ratio = max(base_ratio, args.lr_decay_rate ** math.floor((t - args.lr_warmup_steps) /
                                                                     args.lr_decay_steps))
        opt.param_groups[0]['lr'] = args.learning_rate * ratio
    return opt,  args.learning_rate * ratio


def load_state_dict(model, args, logger):
    logger.info(f'load bert weight: {args.init_checkpoint}')
    state_dict = torch.load(args.init_checkpoint, map_location='cpu')
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    # new_state_dict=state_dict.copy()
    # for kye ,value in state_dict.items():
    #     new_state_dict[kye.replace("bert","c_bert")]=value
    # state_dict=new_state_dict
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            # logger.info("name {} chile {}".format(name,child))
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
    logger.info("missing keys:{}".format(missing_keys))
    logger.info('unexpected keys:{}'.format(unexpected_keys))
    logger.info('error msgs:{}'.format(error_msgs))

    return model


def save_model(model, global_step, args, logging, running_time, benchmark, baseline):
    log_dir_name = running_time
    save_dir = os.path.join(args.output_dir, f'{baseline}', log_dir_name, f'benchmark-{benchmark}')
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir, exist_ok=True)

    # save file
    with open(save_dir + '/{}'.format('config.py'), 'w', encoding='utf-8') as fp1:
        with open(f'{args.root_dir}/config.py', 'r', encoding='utf-8') as fp2:
            fp1.write(fp2.read())
    if os.path.exists(f'{args.root_dir}/train.log'):
        with open(save_dir + '/{}'.format('train.log'), 'w', encoding='utf-8') as fp1:
            with open(f'{args.root_dir}/train.log', 'r', encoding='utf-8') as fp2:
                fp1.write(fp2.read())

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    logging.info(f'save model: {save_dir}/pytorch_model.bin')
    torch.save(model_to_save.state_dict(), save_dir + f'/pytorch_model.bin')


if __name__ == '__main__':
    from config import Config

    args = Config().args
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # multi_scale_cnns = MultiScaleCNN(args)
    # multi_scale_cnns.to(device)
    # input_tensor = torch.randn(size=(16, 256, 768), dtype=torch.float32).to(device)
    # re = multi_scale_cnns(input_tensor)

    one_scale_attention = OneScaleAttentionLayer(args)
    one_scale_attention.to(device)
    q = torch.randn(size=(40, 128, 256), dtype=torch.float32).to(device)
    a = torch.randn(size=(40, 256, 256), dtype=torch.float32).to(device)
    q_attn, a_attn = one_scale_attention(q, a)

    L = MarginLoss(args)
    loss, sim = L(q_attn, a_attn, q_attn, a_attn)