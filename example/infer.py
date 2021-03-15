#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/3/15 10:40 上午
# @File  : infer.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 推理模型

import opennre

def infer_wiki80_cnn_softmax():
    model = opennre.get_model('wiki80_cnn_softmax')
    result = model.infer({
                             'text': 'He was the son of Máel Dúin mac Máele Fithrich, and grandson of the high king Áed Uaridnach (died 612).',
                             'h': {'pos': (18, 46)}, 't': {'pos': (78, 91)}})
    print(result)


def infer_wiki80_bert_softmax():
    """
    有一些错误
    :return:
    """
    model = opennre.get_model('wiki80_bert_softmax')
    result = model.infer({
                             'text': 'He was the son of Máel Dúin mac Máele Fithrich, and grandson of the high king Áed Uaridnach (died 612).',
                             'h': {'pos': (18, 46)}, 't': {'pos': (78, 91)}})
    print(result)


def infer_wiki80_bertentity_softmax():
    model = opennre.get_model('wiki80_bertentity_softmax')
    result = model.infer({
                             'text': 'He was the son of Máel Dúin mac Máele Fithrich, and grandson of the high king Áed Uaridnach (died 612).',
                             'h': {'pos': (18, 46)}, 't': {'pos': (78, 91)}})
    print(result)


def infer_tacred_bertentity_softmax():
    model = opennre.get_model('tacred_bertentity_softmax')
    result = model.infer({
                             'text': 'He was the son of Máel Dúin mac Máele Fithrich, and grandson of the high king Áed Uaridnach (died 612).',
                             'h': {'pos': (18, 46)}, 't': {'pos': (78, 91)}})
    print(result)

def infer_tacred_bert_softmax():
    model = opennre.get_model('tacred_bert_softmax')
    result = model.infer({
                             'text': 'He was the son of Máel Dúin mac Máele Fithrich, and grandson of the high king Áed Uaridnach (died 612).',
                             'h': {'pos': (18, 46)}, 't': {'pos': (78, 91)}})
    print(result)

if __name__ == '__main__':
    infer_wiki80_bert_softmax()
    # infer_tacred_bertentity_softmax()
    # infer_tacred_bert_softmax()