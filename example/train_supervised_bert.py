# coding:utf-8
import torch
import numpy as np
import json
import opennre
from opennre import encoder, model
import sys
import os
import argparse
import logging

def doargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_path', default='bert-base-uncased',
                        help='Pre-trained ckpt path / model name (hugginface)')
    parser.add_argument('--ckpt', default='',
                        help='Checkpoint 位置')
    parser.add_argument('--pooler', default='entity', choices=['cls', 'entity'],
                        help='句子表达，使用cls还是实体的表达')
    parser.add_argument('--do_train', action='store_true',help='训练模型')
    parser.add_argument('--do_test', action='store_true',help='测试模型')
    parser.add_argument('--mask_entity', action='store_true',
                        help='是否mask实体提及')

    # Data
    parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
                        help='选择best checkpoint时使用哪个 Metric')
    parser.add_argument('--dataset', default='none', choices=['none', 'semeval', 'wiki80', 'tacred', 'liter'],
                        help='Dataset. 如果数据集不为none，那么需要指定每个单独的训练文件,否则使用几个专用数据集')
    parser.add_argument('--train_file', default='', type=str, help='训练数据集')
    parser.add_argument('--val_file', default='', type=str, help='验证数据集')
    parser.add_argument('--test_file', default='', type=str, help='测试数据集')
    parser.add_argument('--rel2id_file', default='', type=str, help='关系到id的映射文件')

    # Hyper-parameters
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size')
    parser.add_argument('--lr', default=2e-5, type=float,
                        help='Learning rate')
    parser.add_argument('--max_length', default=128, type=int,
                        help='最大序列长度')
    parser.add_argument('--max_epoch', default=3, type=int,
                        help='最大训练的epoch')

    args = parser.parse_args()
    return args

def load_dataset_and_framework():
    # Some basic settings
    root_path = '.'
    sys.path.append(root_path)
    if not os.path.exists('ckpt'):
        os.mkdir('ckpt')
    if len(args.ckpt) == 0:
        args.ckpt = '{}_{}_{}'.format(args.dataset, args.pretrain_path, args.pooler)
    ckpt = 'ckpt/{}.pth.tar'.format(args.ckpt)

    if args.dataset != 'none':
        opennre.download(args.dataset, root_path=root_path)
        args.train_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_train.txt'.format(args.dataset))
        args.val_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_val.txt'.format(args.dataset))
        args.test_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_test.txt'.format(args.dataset))
        if not os.path.exists(args.test_file):
            logging.warning("Test file {} does not exist! Use val file instead".format(args.test_file))
            args.test_file = args.val_file
        args.rel2id_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_rel2id.json'.format(args.dataset))
        if args.dataset == 'wiki80':
            args.metric = 'acc'
        else:
            args.metric = 'micro_f1'
    else:
        if not (os.path.exists(args.train_file) and os.path.exists(args.val_file) and os.path.exists(
                args.test_file) and os.path.exists(args.rel2id_file)):
            raise Exception('--train_file, --val_file, --test_file and --rel2id_file 没有指定或文件不存在，请检查. 或者指定 --dataset')

    logging.info('参数:')
    for arg in vars(args):
        logging.info('    {}: {}'.format(arg, getattr(args, arg)))

    rel2id = json.load(open(args.rel2id_file))

    # Define the sentence encoder
    if args.pooler == 'entity':
        sentence_encoder = opennre.encoder.BERTEntityEncoder(
            max_length=args.max_length,
            pretrain_path=args.pretrain_path,
            mask_entity=args.mask_entity
        )
    elif args.pooler == 'cls':
        sentence_encoder = opennre.encoder.BERTEncoder(
            max_length=args.max_length,
            pretrain_path=args.pretrain_path,
            mask_entity=args.mask_entity
        )
    else:
        raise NotImplementedError

    # 初始化softmax模型
    model = opennre.model.SoftmaxNN(sentence_encoder, num_class=len(rel2id), rel2id=rel2id)

    # Define the whole training framework
    myframe = opennre.framework.SentenceRE(
        train_path=args.train_file,
        val_path=args.val_file,
        test_path=args.test_file,
        model=model,
        ckpt=ckpt,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        lr=args.lr,
        opt='adamw',
        parallel=False,  # 是否GPU并发
        num_workers=0  # Dataloader的进程数，使用并发时使用
    )
    return myframe, ckpt

def dotrain():
    #训练模型
    myframe.train_model('micro_f1')

def dotest(ckpt):
    #加载训练好的模型，开始测试
    myframe.load_state_dict(torch.load(ckpt)['state_dict'])
    result = myframe.eval_model(myframe.test_loader)

    #打印结果
    logging.info('Test set results:')
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
        dotrain()
    if args.do_test:
        dotest(ckpt)