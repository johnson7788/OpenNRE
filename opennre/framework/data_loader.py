import torch
import torch.utils.data as data
import os, random, json, logging
import numpy as np
import sklearn.metrics

class SentenceREDataset(data.Dataset):
    """
    Sentence-level relation extraction dataset
    """
    def __init__(self, path, rel2id, tokenizer, kwargs):
        """
        Args:
            path: 数据原始文件
            rel2id: dictionary of relation->id mapping, 关系到id的映射字典
            tokenizer: function of tokenizing，初始化的tokenizer
            kwargs: tokenizer的其它参数
        """
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.kwargs = kwargs

        # Load the file
        f = open(path)
        self.data = []
        for line in f.readlines():
            line = line.rstrip()
            if len(line) > 0:
                self.data.append(eval(line))
        f.close()
        logging.info("加载 RE 数据集 {} with {} 行和{} 个关系.".format(path, len(self.data), len(self.rel2id)))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """
        获取一条数据，数据经过tokenizer后的, 会连续获取一个batch_size的数据
        :param index: 47393
        :return:  包含5个元素的列表， 关系的id, 句子id, att_mask, 实体1的起始位置tensor, 实体2的结束位置
        """
        item = self.data[index]
        seq = list(self.tokenizer(item, **self.kwargs))
        # [self.rel2id[item['relation']]] 代表关系的id
        res = [self.rel2id[item['relation']]] + seq
        return res # label, seq1, seq2, ...
    
    def collate_fn(data):
        """
        对一个batch的数据进行处理，里面是一个列表，是batch_size大小，是上面__getitem__返回的每一条数据拼接成的一个batch_size大小的列表,这里每个原素是包含5个元素
        经过list(zip(*data))处理后每个元素的第一维度是batch_size
        :return:
        """
        data = list(zip(*data))
        labels = data[0]
        # seqs是4种特征，句子id, att_mask, 实体1的起始位置tensor, 实体2的结束位置
        seqs = data[1:]
        batch_labels = torch.tensor(labels).long() # (B)
        batch_seqs = []
        for seq in seqs:
            # 把每个特征的batch_size 拼接起来 [1,128] --> [batch_size, max_seq_length] , 然后放到一个列表中
            batch_seqs.append(torch.cat(seq, 0)) # (B, L)
        #返回包含一个列表的含4个元素，每个元素都是tensor
        return [batch_labels] + batch_seqs
    
    def eval(self, pred_result, use_name=False):
        """
        Args:
            pred_result: 预测标签(id)的列表，在生成dataloader时确保`shuffle`参数设置为`False`。
            use_name: 如果True，`pred_result`包含预测的关系名，而不是id。
        Return:
            {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
        """
        correct = 0
        total = len(self.data)
        correct_positive = 0
        pred_positive = 0
        gold_positive = 0
        neg = -1
        for name in ['NA', 'na', 'no_relation', 'Other', 'Others']:
            if name in self.rel2id:
                if use_name:
                    neg = name
                else:
                    neg = self.rel2id[name]
                break
        for i in range(total):
            if use_name:
                golden = self.data[i]['relation']
            else:
                golden = self.rel2id[self.data[i]['relation']]
            if golden == pred_result[i]:
                correct += 1
                if golden != neg:
                    correct_positive += 1
            if golden != neg:
                gold_positive +=1
            if pred_result[i] != neg:
                pred_positive += 1
        acc = float(correct) / float(total)
        try:
            micro_p = float(correct_positive) / float(pred_positive)
        except:
            micro_p = 0
        try:
            micro_r = float(correct_positive) / float(gold_positive)
        except:
            micro_r = 0
        try:
            micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
        except:
            micro_f1 = 0
        result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
        logging.info('评估结果 : {}.'.format(result))
        return result
    
def SentenceRELoader(path, rel2id, tokenizer, batch_size, 
        shuffle, num_workers=8, collate_fn=SentenceREDataset.collate_fn, **kwargs):
    """
    加载数据，返回Dataloader
    :param path:  数据集文件 eg: './benchmark/wiki80/wiki80_train.txt'
    :param rel2id: 关系到id的映射字典
    :param tokenizer: 初始化的tokenizer
    :param batch_size: eg： 16
    :param shuffle: bool
    :param num_workers: 使用的进程数
    :param collate_fn: 数据处理函数
    :param kwargs: tokenizer的其它参数
    :return:
    """
    dataset = SentenceREDataset(path = path, rel2id = rel2id, tokenizer = tokenizer, kwargs=kwargs)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return data_loader

class BagREDataset(data.Dataset):
    """
    Bag-level关系抽取数据集。注意，NA的关系应该命名为 "NA"
    """
    def __init__(self, path, rel2id, tokenizer, entpair_as_bag=False, bag_size=0, mode=None):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
            entpair_as_bag: if True, bags are constructed based on same
                entity pairs instead of same relation facts (ignoring 
                relation labels)
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.entpair_as_bag = entpair_as_bag
        self.bag_size = bag_size

        # Load the file
        f = open(path)
        self.data = []
        for line in f:
            line = line.rstrip()
            if len(line) > 0:
                self.data.append(eval(line))
        f.close()

        #构造bag-level数据集（一个包含共享相同关系事实的实例）。  Construct bag-level dataset (a bag contains instances sharing the same relation fact)
        if mode == None:
            self.weight = np.ones((len(self.rel2id)), dtype=np.float32)
            self.bag_scope = []
            self.name2id = {}
            self.bag_name = []
            self.facts = {}
            for idx, item in enumerate(self.data):
                fact = (item['h']['id'], item['t']['id'], item['relation'])
                if item['relation'] != 'NA':
                    self.facts[fact] = 1
                if entpair_as_bag:
                    name = (item['h']['id'], item['t']['id'])
                else:
                    name = fact
                if name not in self.name2id:
                    self.name2id[name] = len(self.name2id)
                    self.bag_scope.append([])
                    self.bag_name.append(name)
                self.bag_scope[self.name2id[name]].append(idx)
                self.weight[self.rel2id[item['relation']]] += 1.0
            self.weight = 1.0 / (self.weight ** 0.05)
            self.weight = torch.from_numpy(self.weight)
        else:
            pass
  
    def __len__(self):
        return len(self.bag_scope)

    def __getitem__(self, index):
        bag = self.bag_scope[index]
        if self.bag_size > 0:
            if self.bag_size <= len(bag):
                resize_bag = random.sample(bag, self.bag_size)
            else:
                resize_bag = bag + list(np.random.choice(bag, self.bag_size - len(bag)))
            bag = resize_bag
            
        seqs = None
        rel = self.rel2id[self.data[bag[0]]['relation']]
        for sent_id in bag:
            item = self.data[sent_id]
            seq = list(self.tokenizer(item))
            if seqs is None:
                seqs = []
                for i in range(len(seq)):
                    seqs.append([])
            for i in range(len(seq)):
                seqs[i].append(seq[i])
        for i in range(len(seqs)):   #seqs 里面包含4个特征，每个特征是一个bag数量的元素，每个元素的维度是1，seq_length
            seqs[i] = torch.cat(seqs[i], 0) # (n, L), 拼接后，seqs还是4个元素，每个元素的维度变成(bag_size, seq_length)
        # 返回, rel: 0 关系的id， 关系的名称: ('T51', 'T52', 'Family'),  len(bag): bag_size, seqs: (bag_size, seq_length)
        return [rel, self.bag_name[index], len(bag)] + seqs
  
    def collate_fn(data):
        data = list(zip(*data))
        label, bag_name, count = data[:3]
        seqs = data[3:]
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0) # (sumn, L)
            seqs[i] = seqs[i].expand((torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1, ) + seqs[i].size())
        scope = [] # (B, 2)
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        assert(start == seqs[0].size(1))
        scope = torch.tensor(scope).long()
        label = torch.tensor(label).long() # (B)
        return [label, bag_name, scope] + seqs

    def collate_bag_size_fn(data):
        data = list(zip(*data))
        # label: 一个batch的列表 (0, 5, 3, 3, 3, 5, 3, 4), bag_name: Batch_size大小，每个元素类似 ('T223', 'T210', 'Family')， count: 一个batch中每个bag_size大小 (60, 60, 60, 60, 60, 60, 60, 60)
        label, bag_name, count = data[:3]
        #只要text_id, pos1_id, pos2_id, mask的id先不要
        seqs = data[3:]
        for i in range(len(seqs)):
            seqs[i] = torch.stack(seqs[i], 0) # (batch, bag, L), seqs里的每个元素维度变成 [batch_size, bag_size, seq_length]
        scope = [] # (B, 2)
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        label = torch.tensor(label).long() # (B)
        # scope 是每个bag的个数的边界记录
        return [label, bag_name, scope] + seqs

  
    def eval(self, pred_result):
        """
        Args:
            pred_result: a list with dict {'entpair': (head_id, tail_id), 'relation': rel, 'score': score}.
                Note that relation of NA should be excluded.
        Return:
            {'prec': narray[...], 'rec': narray[...], 'mean_prec': xx, 'f1': xx, 'auc': xx}
                prec (precision) and rec (recall) are in micro style.
                prec (precision) and rec (recall) are sorted in the decreasing order of the score.
                f1 is the max f1 score of those precison-recall points
        """
        # sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
        prec = []
        rec = []
        correct = 0
        total = len(pred_result)
        for i, item in enumerate(pred_result):
            entity1, entity2, predict, score, groud_truth = item['entpair'][0], item['entpair'][1], item['predict'], item['score'], item['groud_truth']
            if predict == groud_truth:
                correct += 1
            # prec.append(float(correct) / float(i + 1))
            # rec.append(float(correct) / float(total))
        #准确率好像有问题，计算时entpair是不需要提供的
        acc = float(correct) / float(total)
        # auc = sklearn.metrics.auc(x=rec, y=prec)
        # np_prec = np.array(prec)
        # np_rec = np.array(rec)
        # f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
        # mean_prec = np_prec.mean()
        # return {'micro_p': np_prec, 'micro_r': np_rec, 'micro_p_mean': mean_prec, 'micro_f1': f1, 'auc': auc, 'acc': acc}
        return {'acc': acc}

def BagRELoader(path, rel2id, tokenizer, batch_size, 
        shuffle, entpair_as_bag=False, bag_size=0, num_workers=8, 
        collate_fn=BagREDataset.collate_fn):
    if bag_size == 0:
        collate_fn = BagREDataset.collate_fn
    else:
        collate_fn = BagREDataset.collate_bag_size_fn
    dataset = BagREDataset(path, rel2id, tokenizer, entpair_as_bag=entpair_as_bag, bag_size=bag_size)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return data_loader

if __name__ == '__main__':
    pred_result = "hello"