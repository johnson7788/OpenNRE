import os, logging, json
from tqdm import tqdm
import torch
from torch import nn, optim
from .data_loader import SentenceRELoader
from .utils import AverageMeter

class SentenceRE(nn.Module):
    """
    模型训练函数
    """
    def __init__(self, 
                 model,   #初始化后的模型
                 train_path, #训练文件路径
                 val_path, 
                 test_path,
                 ckpt,    #要保存的模型的checkpoint路径
                 batch_size=32, 
                 max_epoch=100, 
                 lr=0.1, 
                 weight_decay=1e-5, 
                 warmup_step=300,
                 opt='sgd',
                 parallel = False,     #是否使用并发的GPU
                 num_workers = 0    # Dataloader 的并发数,只主进程加载
                 ):
    
        super().__init__()
        self.max_epoch = max_epoch
        # Load data
        if train_path != None:
            self.train_loader = SentenceRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                shuffle=True, num_workers=num_workers)

        if val_path != None:
            self.val_loader = SentenceRELoader(
                val_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                shuffle=False, num_workers=num_workers)
        
        if test_path != None:
            self.test_loader = SentenceRELoader(
                test_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                shuffle=False, num_workers=num_workers
            )
        # Model
        self.model = model
        if parallel:
            self.parallel_model = nn.DataParallel(self.model)
        else:
            self.parallel_model = None
        # Criterion
        self.criterion = nn.CrossEntropyLoss()
        # Params and optimizer
        params = self.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw': # Optimizer for BERT
            from transformers import AdamW
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("无效优化器. Must be 'sgd' or 'adam' or 'adamw'.")
        # Warmup
        if warmup_step > 0:
            from transformers import get_linear_schedule_with_warmup
            training_steps = self.train_loader.dataset.__len__() // batch_size * self.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step, num_training_steps=training_steps)
        else:
            self.scheduler = None
        # Cuda
        if torch.cuda.is_available():
            logging.info("检测到GPU可用，使用GPU")
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def train_model(self, metric='acc'):
        """
        训练模型
        :param metric: 使用哪个指标作为checkpoint的最佳指标,支持acc， micro_p， micro_r， micro_f1
        :return:
        """
        best_metric = 0
        global_step = 0
        for epoch in range(self.max_epoch):
            self.train()
            logging.info("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            t = tqdm(self.train_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                #  一个batch_size的label, 16
                label = data[0]
                # args是一个包含4个元素的列表， [token ids, atten_mask, entity1_start_id, entity2_end_id]
                args = data[1:]
                # args是一个包含4个元素的列表作为特征放入模型， logits shape [batch_size, num_classes]
                if self.parallel_model:
                    logits = self.parallel_model(*args)
                else:
                    logits = self.model(*args)
                loss = self.criterion(logits, label)
                score, pred = logits.max(-1) # (B)
                acc = float((pred == label).long().sum()) / label.size(0)
                #记录日志
                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)
                # Optimize
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1
            # Val 
            logging.info("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader) 
            logging.info('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
            if result[metric] > best_metric:
                logging.info("Best ckpt and saved.")
                folder_path = '/'.join(self.ckpt.split('/')[:-1])
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
                best_metric = result[metric]
        logging.info("Best %s on val set: %f" % (metric, best_metric))

    def eval_model(self, eval_loader):
        """
        评估模型
        :param eval_loader: 评估数据集
        :return:
        """
        self.eval()
        avg_acc = AverageMeter()
        pred_result = []
        with torch.no_grad():
            t = tqdm(eval_loader, desc='评估: ')
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                args = data[1:]
                if self.parallel_model:
                    logits = self.parallel_model(*args)
                else:
                    logits = self.model(*args)
                score, pred = logits.max(-1) # (B)
                # Save result
                for i in range(pred.size(0)):
                    pred_result.append(pred[i].item())
                # Log
                acc = float((pred == label).long().sum()) / label.size(0)
                avg_acc.update(acc, pred.size(0))
                t.set_postfix(acc=avg_acc.avg)
        result = eval_loader.dataset.eval(pred_result)
        return result

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

