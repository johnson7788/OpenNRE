import logging
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from .base_encoder import BaseEncoder

class BERTEncoder(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768
        self.mask_entity = mask_entity
        logging.info(f'加载预训练的 BERT pre-trained checkpoint: {pretrain_path}')
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)

    def forward(self, token, att_mask):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        _, x = self.bert(token, attention_mask=att_mask, return_dict=False)
        return x

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False
        
        if not is_token:
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        else:
            sent0 = self.tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))
            ent0 = self.tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))
            sent1 = self.tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))
            ent1 = self.tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))
            sent2 = self.tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))

        if self.mask_entity:
            ent0 = ['[unused4]'] if not rev else ['[unused5]']
            ent1 = ['[unused5]'] if not rev else ['[unused4]']
        else:
            ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
            ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']

        re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
        
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask


class BERTEntityEncoder(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        """
        加载huggface的预训练模型，初始化tokenizer
        Args:
            max_length: 最大序列长度
            pretrain_path: 预训练模型的路径
            blank_padding: bool
            mask_entity: bool, 是否mask掉实体，如果mask掉实体，那么实体的名称就会用特殊的字符表示，但是模型性能会下降
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768 * 2
        self.mask_entity = mask_entity
        logging.info(f'加载预训练的 BERT pre-trained checkpoint: {pretrain_path}')
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, token, att_mask, pos1, pos2):
        """
        Args:
            token: (Batch_size, seq_length), index of tokens, 句子和实体拼接后的token
            att_mask: (Batch_size, seq_length), attention mask (1 for contents and 0 for padding)
            pos1: (Batch_size, 1), position of the head entity starter, [batch_size, 1] 实体1的开始位置
            pos2: (Batch_size, 1), position of the tail entity starter, [batch_size, 1] 实体2的结束位置
        Return:
            (B, 2H), representations for sentences
        """
        # hidden [batch_size, seq_len, output_demision], 先经过bert
        hidden, _ = self.bert(token, attention_mask=att_mask, return_dict=False,)
        # 初始化一个向量 onehot_head shape [batch_size, seq_len]
        onehot_head = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_tail = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        #获取实体位置的向量,
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        #head_hidden,tail_hidden --> [batch_size, hidden_demision], [16,768]
        head_hidden = (onehot_head.unsqueeze(2) * hidden).sum(1)  # (B, H)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden).sum(1)  # (B, H)
        #把实体的头和尾拼接起来, x shape  (Batch_size, 2*hidden_demision), 放入线性模型
        x = torch.cat([head_hidden, tail_hidden], 1)
        x = self.linear(x)
        return x

    def tokenize(self, item):
        """
        Args:
            item: 一条数据，包括 'text' 或 'token', 'h' and 't'， eg: {'token': ['It', 'then', 'enters', 'the', 'spectacular', 'Clydach', 'Gorge', ',', 'dropping', 'about', 'to', 'Gilwern', 'and', 'its', 'confluence', 'with', 'the', 'River', 'Usk', 'Ordnance', 'Survey', 'Explorer', 'map', 'OL13', ',', '"', 'Brecon', 'Beacons', 'National', 'Park', ':', 'eastern', 'area', '"', '.'], 'h': {'name': 'gilwern', 'id': 'Q5562649', 'pos': [11, 12]}, 't': {'name': 'river usk', 'id': 'Q19699', 'pos': [17, 19]}, 'relation': 'located in or next to body of water'}
        Return:
             indexed_tokens，整理好的id, att_mask对应的mask, pos1实体1的起始位置, pos2实体2的结束位置
        """
        # Sentence -> token, text表示文本没有进行token，是一个完整的句子，token表示文本token过了，是一个列表了
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True
        # pos_head 第一个实体的位置 eg: [11, 12]， pos_tail第二个实体的位置
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']
        pos_min = pos_head
        pos_max = pos_tail
        #确定哪个实体在句子的前面和后面
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False
        
        if not is_token:
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        else:
            #sent0 是句子到第一个实体的单词的位置， eg: ['it', 'then', 'enters', 'the', 'spectacular', 'cl', '##yd', '##ach', 'gorge', ',', 'dropping', 'about', 'to']
            sent0 = self.tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))
            # ent0实体的tokenizer, eg: ['gil', '##wer', '##n']
            ent0 = self.tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))
            # sent1是第一个实体到第二个实体之间的句子, eg: ['and', 'its', 'confluence', 'with', 'the']
            sent1 = self.tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))
            #是第二个实体, eg: ['river', 'us', '##k']
            ent1 = self.tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))
            #第二个实体到句子末尾 eg:  ['ordnance', 'survey', 'explorer', 'map', 'ol', '##13', ',', '"', 'br', '##ec', '##on', 'beacon', '##s', 'national', 'park', ':', 'eastern', 'area', '"', '.']
            sent2 = self.tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))

        if self.mask_entity:
            ent0 = ['[unused4]'] if not rev else ['[unused5]']
            ent1 = ['[unused5]'] if not rev else ['[unused4]']
        else:
            # 不mask实体，用特殊的mask围住实体
            ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
            ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']
        #在句子开头和末尾加上special token
        re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
        # 实体的位置也改变了
        pos1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)
        pos2 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)
        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)
        #把token转换成id
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Position， pos1 eg: tensor([[2]])
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()

        # Padding, 如果少于长度，开始padding,多余长度，截断
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # shape,[1, max_seq_length] torch.Size([1, 128])

        # Attention mask, 只有真实的长度为1，其它的地方为0
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask, pos1, pos2
