import argparse
import os
class Config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_config_file", default=None, type=str, required=False,
                        help="The config json file corresponding to the pre-trained BERT model. "
                             "This specifies the model architecture.")
    parser.add_argument("--vocab_file", default=None, type=str, required=False,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--init_checkpoint", default=None, type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model checkpoints and predictions will be written.")

    parser.add_argument("--train_text_data", default=None, type=str,
                        help="")
    parser.add_argument("--dev_text_data", default=None, type=str,
                        help="")
    parser.add_argument("--test_text_data", default=None, type=str,
                        help="")

    parser.add_argument("--gpu_ids", default=None, type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument('--seed', type=int, default=41, help="random seed for initialization")

    #train
    parser.add_argument("--batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--num_train_epochs", default=15, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--dev_step", default=100, type=float,
                        help="Total number of training epochs to perform.") #500

    # optimizer
    parser.add_argument("--learning_rate", default=0.01, type=float, help="The initial learning rate for Adam.")

    # gru
    parser.add_argument("--gru_hidden_size", default=150, type=int, help="")

    # multi-scale cnn
    parser.add_argument("--cnn_kernel_num", default=150, type=int, help="The initial learning rate for Adam.")

    # loss:
    parser.add_argument("--margin", default=0.1, type=float,
                        help="")
    parser.add_argument("--eps", default=1e-06, type=float,
                        help="")
    parser.add_argument("--reduction", default='mean', type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument('--max_grad_norm', default=5.0, type=float)


    args = parser.parse_args()
    args.root_dir = os.path.abspath('./')
    args.before_dir = os.path.abspath('../')
    # setting gpu
    args.gpu_ids = "0"
    # setting file paths
    args.train_text_data = args.before_dir + '/gen_data/data/train_text_data.pkl'
    args.dev_text_data = args.before_dir + '/gen_data/data/dev_text_data.pkl'
    args.test_text_data = args.before_dir + '/gen_data/data/test_text_data.pkl'
    args.vocab_file = args.before_dir + '/pretrain_bert_6/chinese_wwm_pytorch/vocab.txt'
    args.bert_config_file = args.before_dir + '/pretrain_bert_6/6_layers_bert_config.json'
    #args.init_checkpoint = args.before_dir + '/pretrain_bert_6/output_model/2019-10-14@09_02_15/pretrain_bert_pytorch_model_10.bin'
    args.output_dir = args.root_dir + '/output_model'
    # setting data paramsoutput_model_c300
    args.max_seq_length = 256
    # w2v
    args.w2v_vocab2id = args.before_dir + '/trainW2v/model/w2v_iter20000_size300/vocab2id.pkl'
    args.w2v_weight = args.before_dir + '/trainW2v/model/w2v_iter20000_size300/vocab_weight.pkl'

