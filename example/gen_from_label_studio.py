#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/3/29 9:48 上午
# @File  : gen_from_label_studio.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 从label_studio获取生成的数据

import glob
import os
import json
import random
import re
import collections
import pandas as pd
import matplotlib.pyplot as plt
import requests

def collect_json(dirpath):
    """
    收集目录下的所有json文件，合成一个大的列表
    :param dirpath:如果是目录，那么搜索所有json文件，如果是文件，那么直接读取文件
    :return: 返回列表
    """
    #所有文件的读取结果
    data = []
    if os.path.isdir(dirpath):
        search_file = os.path.join(dirpath, "*.json")
        # 搜索所有json文件
        json_files = glob.glob(search_file)
    else:
        json_files = [dirpath]
    for file in json_files:
        with open(file, 'r') as f:
            file_data = json.load(f)
            print(f"{file}中包含数据{len(file_data)} 条")
            data.extend(file_data)
    print(f"共收集数据{len(data)} 条")
    return data

def format_data(data, keep_cancel=False):
    """
    整理数据，只保留需要的字段
    :param data:,
    :param keep_cancel: 是否返回cancel的数据，如果是True，那么返回data, cancel_data，否则返回data
    :return:[(text, keyword, start_idx, end_idx, label,channel,wordtype)]
    """
    newdata = []
    #未标注，取消标注的数据
    cancel_data = []
    #多人标注的条数, 一共多人标注的条数
    repeats_completions = 0
    #多人标注时，不一致的数量
    not_same_label_num = 0
    for one in data:
        text = one['data']['text']
        keyword = one['data']['keyword']
        channel = one['data']['channel']
        wordtype= one['data']['wordtype']
        if not one['completions'][0]['result']:
            #过滤掉有问题的数据，有的数据是有问题的，所以没有标记，过滤掉
            continue
        # 校验一下一共几个关键字，标注了几个
        research = re.findall(keyword,text)
        if len(research) > 1:
            annotation_num = len(one['completions'][0]['result'])
            if len(research) != annotation_num:
                print("标注的数量不匹配, 不会造成影响，例如句子中有2个词，我们只标注了一个词")
        #一句话中的多个标注结果
        #如果有多个标注结果，对比下标注结果，把不同的打印出来,
        canceled = []
        # if len(one['completions']) >1:
        # print(f"标注结果大于1个，打印出来，只打印出2个人标注完全不一样的结果")
        unique = {}
        #存储不被取消和没有重复的标注的数据
        results = []
        #多人标注数据统计
        if len(one['completions']) >1:
            repeats_completions +=1
        for completion in one['completions']:
            if completion.get("was_cancelled"):
                canceled.extend(completion['result'])
                continue
            for res in completion['result']:
                start_idx = res['value']['start']
                end_idx = res['value']['end']
                label = res['value']['labels'][0]
                unique_sentence_idx = f"{str(start_idx):{str(end_idx)}}"
                unique_sentence_label= f"{label}:{keyword}"
                if unique_sentence_idx in unique.keys():
                    if unique_sentence_label != unique[unique_sentence_idx]:
                        print(f"2个人标注的数据不一致, 句子为:{text}   标注的单词位置: {start_idx}-{end_idx}   一个标注为:{unique[unique_sentence_idx]}  另一个标注为:{unique_sentence_label},标注不一致的被丢弃")
                        not_same_label_num += 1
                    else:
                        # 2个人标注的一致,忽略
                        print(f"2个人标注的数据一致,标注了多次,句子为:{text}   标注的单词位置: {start_idx}-{end_idx}   标注为:{unique[unique_sentence_idx]}")
                        continue
                else:
                    unique[unique_sentence_idx] = unique_sentence_label
                    results.append(res)
        if canceled:
            #如果canceled有数据，那么整个数据都不要了，直接cancel, 那么我们保留一个cancel数据就可以了
            res = canceled[0]
            start_idx = res['value']['start']
            end_idx = res['value']['end']
            label = "CANCELLED"
            new = (text, keyword, start_idx, end_idx, label, channel, wordtype)
            cancel_data.append(new)
        else:
            for res in results:
                start_idx = res['value']['start']
                end_idx = res['value']['end']
                label = res['value']['labels'][0]
                new = (text,keyword,start_idx,end_idx,label,channel,wordtype)
                newdata.append(new)
    print(f"处理完成后的数据总数是{len(newdata)}, 存在{repeats_completions}条多人标注的数据, 标注不一致的数据有{not_same_label_num}条, 被跳过的数据有{len(cancel_data)}")
    if keep_cancel:
        return newdata, cancel_data
    else:
        return newdata

def analysis_data(data):
    """
    wandb分析数据, 基本文本长度在1000左右，关系数在10-30左右
    :param data:
    :return:
    """
    import wandb
    wandb.init(project='relations')
    # positive_labels 保存所有正样本
    for idx, one in enumerate(data):
        print(f"第{idx}条数据")
        text = one['data']['text']
        text_length = len(text)
        #品牌，用逗号分隔
        # brand = one['data']['brand']
        channel = one['data']['channel']
        #需求，用逗号分隔
        # requirement = one['data']['requirement']
        #标记完成的数量
        result = one['completions'][0]['result']
        if not result:
            #过滤掉有问题的数据，有的数据是有问题的，所以没有标记，过滤掉
            continue
        completetion_num = len(result)
        #标注的耗时
        brands_num = sum( 1 for i in result if i.get('value', {'labels':['other']})['labels'][0] == '品牌')
        requires_num = sum( 1 for i in result if i.get('value', {'labels':['other']})['labels'][0] == '需求')
        relations_num = sum( 1 for i in result if i.get('type') == 'relation')
        postive_num = sum(1 for i in result if i.get('type') == 'relation' and i['labels'][0] == '是')
        negitive_num = sum(1 for i in result if i.get('type') == 'relation' and i['labels'][0] == '否')
        wandb.log({
            "idx": idx,
            "text_length": text_length,
            "completetion_num":completetion_num,
            "brands_num": brands_num,
            "requires_num": requires_num,
            "relations_num":relations_num,
            "postive_num": postive_num,
            "negitive_num": negitive_num
        })
    print(f"上传到wandb完成")


if __name__ == '__main__':
    data = collect_json(dirpath="/opt/lavector/relation")
