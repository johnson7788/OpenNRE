# coding:utf-8
import torch
import numpy as np
import json
import opennre
from opennre import encoder, model, framework
import sys
import os
import argparse
import logging

def doargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='', help='Checkpoint 名字')
    parser.add_argument('--do_train', action='store_true', help='训练模型')
    parser.add_argument('--do_test', action='store_true', help='测试模型')
    # Data
    parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'], help='选择best checkpoint时使用哪个 Metric')
    parser.add_argument('--dataset', default='wiki80', choices=['none', 'semeval', 'wiki80', 'tacred'],
            help='Dataset. 如果数据集不为none，那么需要指定每个单独的训练文件,否则使用几个专用数据集')
    parser.add_argument('--train_file', default='', type=str, help='训练数据集')
    parser.add_argument('--val_file', default='', type=str,help='验证数据集')
    parser.add_argument('--test_file', default='', type=str, help='测试数据集')
    parser.add_argument('--rel2id_file', default='', type=str,help='关系到id的映射文件')

    # Hyper-parameters
    parser.add_argument('--batch_size', default=32, type=int,
            help='Batch size')
    parser.add_argument('--lr', default=1e-1, type=float,
            help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
            help='Weight decay')
    parser.add_argument('--max_length', default=128, type=int,
                        help='最大序列长度')
    parser.add_argument('--max_epoch', default=100, type=int,
                        help='最大训练的epoch')

    args = parser.parse_args()
    return args

def load_dataset_and_framework():
    root_path = '.'
    sys.path.append(root_path)
    if not os.path.exists('ckpt'):
        os.mkdir('ckpt')
    if len(args.ckpt) == 0:
        args.ckpt = '{}_{}'.format(args.dataset, 'cnn')
    ckpt = 'ckpt/{}.pth.tar'.format(args.ckpt)
    if args.dataset != 'none':
        opennre.download(args.dataset, root_path=root_path)
        args.train_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_train.txt'.format(args.dataset))
        args.val_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_val.txt'.format(args.dataset))
        args.test_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_test.txt'.format(args.dataset))
        args.rel2id_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_rel2id.json'.format(args.dataset))
        if args.dataset == 'wiki80':
            args.metric = 'acc'
        else:
            args.metric = 'micro_f1'
    else:
        if not (os.path.exists(args.train_file) and os.path.exists(args.val_file) and os.path.exists(args.test_file) and os.path.exists(args.rel2id_file)):
            raise Exception('--train_file, --val_file, --test_file and --rel2id_file are not specified or files do not exist. Or specify --dataset')

    logging.info('Arguments:')
    for arg in vars(args):
        logging.info('    {}: {}'.format(arg, getattr(args, arg)))

    rel2id = json.load(open(args.rel2id_file))

    # Download glove
    opennre.download('glove', root_path=root_path)
    word2id = json.load(open(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_word2id.json')))
    word2vec = np.load(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_mat.npy'))

    # Define the sentence encoder
    sentence_encoder = opennre.encoder.CNNEncoder(
        token2id=word2id,
        max_length=args.max_length,
        word_size=50,
        position_size=5,
        hidden_size=230,
        blank_padding=True,
        kernel_size=3,
        padding_size=1,
        word2vec=word2vec,
        dropout=0.5
    )


    # Define the model
    model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

    # Define the whole training framework
    framework = opennre.framework.SentenceRE(
        train_path=args.train_file,
        val_path=args.val_file,
        test_path=args.test_file,
        model=model,
        ckpt=ckpt,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        opt='sgd'
    )
    return myframe, ckpt

def dotrain(metric):
    """
    使用哪个metric检测checkpoint，并保存
    :param metric:
    :return:
    """
    #训练模型
    myframe.train_model(metric)

def dotest(ckpt):
    #加载训练好的模型，开始测试
    myframe.load_state_dict(torch.load(ckpt)['state_dict'])
    result = myframe.eval_model(myframe.test_loader)

    #打印结果
    logging.info('测试集结果:')
    logging.info('Accuracy: {}'.format(result['acc']))
    logging.info('Micro precision: {}'.format(result['micro_p']))
    logging.info('Micro recall: {}'.format(result['micro_r']))
    logging.info('Micro F1: {}'.format(result['micro_f1']))

if __name__ == '__main__':
    args = doargs()
    logfile = "train.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logfile, mode='w', encoding='utf-8'),
        ]
    )
    myframe, ckpt = load_dataset_and_framework()
    if args.do_train:
        dotrain(args.metric)
    if args.do_test:
        dotest(ckpt)