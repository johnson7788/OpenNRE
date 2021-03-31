#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/3/19 10:58 上午
# @File  : gen_chinese_data.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 根据Chinese-Literature-NER-RE-Dataset提供的数据格式，生成我们需要的训练数据格式
# 由于Chinese-Literature-NER-RE-Dataset是文档级的数据，所以其实需要更高效的训练和预测方法，例如图卷积的网络
import os
import json
import re
import random

def gen_rel2id(train_dir, destination='/Users/admin/git/OpenNRE/benchmark/liter/liter_rel2id.json'):
    """
    根据Chinese-Literature-NER-RE-Dataset的训练目录生成关系到id的映射
    :param train_dir: *.ann和*.txt结尾的文件
    :param destination: 输出的目标json文件
    :return:
    """
    relations = []
    files = os.listdir(train_dir)
    #过滤出标注的文件
    files = [f for f in files if f.endswith('.ann')]
    for file in files:
        annfile = os.path.join(train_dir,file)
        with open(annfile, 'r') as f:
            for line in f:
                if line.startswith('R'):
                    line = line.strip()
                    line_split = re.split('[\t ]', line)
                    relation = line_split[1]
                    if relation == 'Coreference':
                        print(f"文件{annfile}，行 {line}是有问题的")
                    if relation not in relations:
                        print(f'加入关系: {relation}')
                        relations.append(relation)
    desdir = os.path.dirname(destination)
    if not os.path.exists(desdir):
        os.makedirs(desdir)
    assert len(relations) == 9, "关系必须是9个才对"
    rel2id = {rel:idx for idx, rel in enumerate(relations)}
    with open(destination, 'w', encoding='utf-8') as f:
        json.dump(rel2id, f)

def gen_data(source_dir, des_dir, mini_data = False, truncate=-1):
    """
    根据原始目录生成目标训练或测试等文件
    :param source_dir: eg: /Users/admin/git/Chinese-Literature-NER-RE-Dataset/relation_extraction/Training
    :param des_dir:  eg: /Users/admin/git/OpenNRE/benchmark/liter
    :param truncate: -1表示不截断，否则截断，保留截断的最大长度，如果2个实体之间的距离超过了最大长度truncate,那么保留全部
    :return:
    """
    #保存处理好的数据
    data = []
    files = os.listdir(source_dir)
    # 过滤出标注的文件
    ann_files = [f for f in files if f.endswith('.ann')]
    text_files = [f for f in files if f.endswith('.txt')]
    #转出成不带文件后缀的key和文件名为value的字典
    ann_file_dict = {f.split('.')[0]:f for f in ann_files}
    text_file_dict = {f.split('.')[0]: f for f in text_files}
    #如果做了截断，统计下2个实体之间的距离超过了最大的长度，那么没法截断
    longer_num = 0
    for k, v in ann_file_dict.items():
        if text_file_dict.get(k) is None:
            print(f"文件{v} 不存在对应的txt文件，错误")
            continue
        #开始读取ann 文件
        annfile = os.path.join(source_dir, v)
        text_name = text_file_dict.get(k)
        textfile = os.path.join(source_dir, text_name)
        with open(textfile, 'r') as f:
            text = ""
            text_len = []
            for line in f:
                text_len.append(len(line))
                if len(line) == 61:
                    #固定的行长度是61
                    line = line.strip()
                text += line
            # text = f.read()
        #保存所有实体
        entities = []
        #保存所有关系
        rels = []
        with open(annfile, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('R'):
                    line_split = re.split('[\t ]', line)
                    assert len(line_split) == 4, f"关系{annfile}的行 {line}不为4项"
                    rels.append(line_split)
                if line.startswith('T'):
                    line_split = re.split('[\t ]', line)
                    if len(line_split) == 7:
                        # 如果不为5，那么是有逗号隔开的，例如  T81	Metric 539 540;541 542	百 鸟
                        # 只需要T81	Metric 539 540 百
                        pos_stop = line_split[3].split(';')[0]
                        line_split = line_split[:3] + [pos_stop] + [line_split[5]]
                    elif len(line_split) == 5:
                        pass
                    else:
                        raise Exception(f"实体 {annfile} 的行 {line} 不为5项或者7项，有问题，请检查")
                    #把实体的索引，进行减法，因为每61个字符一行，我们去掉了一部分'\n'，所以做减法
                    pos_start = int(line_split[2])
                    pos_stop = int(line_split[3])
                    if pos_start > 61:
                        pos_remind1 = pos_start // 61
                        pos_start = pos_start -pos_remind1
                    if pos_stop > 61:
                        pos_remind2 = pos_stop //61
                        pos_stop = pos_stop - pos_remind2
                    line_split = line_split[:2] + [pos_start, pos_stop] + [line_split[-1]]
                    entities.append(line_split)
        #检查实体, 保存成实体id：实体的type，实体start_idx, 实体stop_idx，实体的值
        ent_dict = {}
        for entity in entities:
            entity_id = entity[0]
            if ent_dict.get(entity_id) is not None:
                print(f"{annfile}: 实体id已经存在过了，冲突的id，请检查 {entity}")
            ent_dict[entity_id] = entity[1:]

        #开始分析所有关系
        for idx, rel in enumerate(rels):
            relation = rel[1]
            arg1, h1_entityid = rel[2].split(':')
            assert arg1 == 'Arg1', f"{rel}分隔的首个字符不是Arg1"
            #实体1的id处理
            h1_entity = ent_dict.get(h1_entityid)
            if h1_entity is None:
                print(f"关系{rel}中对应的实体id{h1_entityid}是不存在的，请检查")
            h1_type,h1_pos_start, h1_pos_stop, h1_entity_value = h1_entity
            h1_pos_start = int(h1_pos_start)
            h1_pos_stop = int(h1_pos_stop)
            arg2, h2_entityid = rel[3].split(':')
            assert arg2 == 'Arg2', f"{rel}分隔的首个字符不是Arg2"
            #实体2的id处理
            h2_entity = ent_dict.get(h2_entityid)
            if h2_entity is None:
                print(f"关系{rel}中对应的实体id{h2_entityid}是不存在的，请检查")
            h2_type, h2_pos_start, h2_pos_stop, h2_entity_value = h2_entity
            h2_pos_start = int(h2_pos_start)
            h2_pos_stop = int(h2_pos_stop)
            # 检查关键字的位置是否匹配
            def get_true_pos(text, value, pos1, pos2, rnum=16):
                #从上下加8个字符获取真实的位置
                index_true_text = text[pos1-rnum:pos2+rnum]
                print(f"实体1: {value}位置不匹配, 上下的2个位置是: {index_true_text}，尝试修复")
                newpos1, newpos2 = pos1, pos2
                if value in index_true_text:
                    sres = re.finditer(re.escape(value), text)
                    for sv in sres:
                        if sv.start() > pos1-rnum and sv.end() < pos2+rnum:
                            newpos1, newpos2 = sv.start(), sv.end()
                            break
                    else:
                        print("通过正则没有匹配到，请检查，用最后一个位置作为索引")
                        newpos1, newpos2 = sv.start(), sv.end()
                else:
                    print("上下浮动了16个，仍然没有匹配，请检查")
                    sres = re.finditer(re.escape(value), text)
                    min_dist = 100
                    for sv in sres:
                        min_dist = min(min_dist, sv.start() - pos1, sv.end() - pos2)
                        if min_dist in [sv.start() - pos1, sv.end() - pos2]:
                            newpos1, newpos2 = sv.start(), sv.end()
                if text[newpos1:newpos2] != value:
                    assert text[newpos1:newpos2] == value, "仍然是匹配错误的位置，请检查"
                return newpos1, newpos2
            # 验证下文本中的实体在文档中的位置时正确的
            if text[h1_pos_start:h1_pos_stop] != h1_entity_value:
                h1_pos_start, h1_pos_stop = get_true_pos(text=text,value=h1_entity_value, pos1=h1_pos_start, pos2=h1_pos_stop)
            if text[h2_pos_start:h2_pos_stop] != h2_entity_value:
                h2_pos_start, h2_pos_stop = get_true_pos(text=text,value=h2_entity_value, pos1=h2_pos_start, pos2=h2_pos_stop)

            if truncate != -1:
                if abs(h1_pos_start - h2_pos_stop) > truncate:
                    print(f'2个实体间的距离很大,超过了{truncate}长度, 不作处理')
                    longer_num +=1
                else:
                    #开始截断数据, 只保留最大长度
                    add_length = truncate - abs(h1_pos_start - h2_pos_stop)
                    added = int(add_length/2)
                    if h1_pos_start < h2_pos_stop:
                        truncate_start = h1_pos_start - added
                        truncate_end = h2_pos_stop + added
                    else:
                        # 说明h2实体在h1实体的前面，那么
                        truncate_start = h2_pos_start - added
                        truncate_end = h1_pos_stop + added
                    if truncate_start <0:
                        truncate_start = 0
                    truncate_text = text[truncate_start:truncate_end]
                    #截断之后需要更新下实体的位置
                    h1_pos_start = h1_pos_start - truncate_start
                    h1_pos_stop = h1_pos_stop - truncate_start
                    h2_pos_start = h2_pos_start - truncate_start
                    h2_pos_stop = h2_pos_stop - truncate_start
                    assert truncate_text[h1_pos_start:h1_pos_stop] == h1_entity_value, f"文件{v}: 索引为 {idx}: 截断后实体的位置和值不匹配"
                    assert truncate_text[h2_pos_start:h2_pos_stop] == h2_entity_value, f"文件{v}: 索引为 {idx}: 截断后实体的位置和值不匹配"
            else:
                truncate_text = text
            # 开始整理成一条数据
            one_data = {
                'text': truncate_text,
                'h': {
                    'name': h1_entity_value,
                    'id': h1_entityid,
                    'pos': [h1_pos_start, h1_pos_stop]
                },
                't': {
                    'name': h2_entity_value,
                    'id': h2_entityid,
                    'pos': [h2_pos_start, h2_pos_stop]
                },
                'relation': relation
            }

            data.append(one_data)
    train_file = os.path.join(des_dir, 'liter_train.txt')
    dev_file = os.path.join(des_dir, 'liter_test.txt')
    test_file = os.path.join(des_dir, 'liter_val.txt')
    print(f"一共处理了{len(ann_files)}个文件，生成{len(data)}条数据,其中2个实体之间超过最大长度的数据有{longer_num}条")
    random.shuffle(data)
    train_num = int(len(data) * 0.8)
    dev_num = int(len(data) * 0.1)
    train_data = data[:train_num]
    dev_data = data[train_num:train_num+dev_num]
    test_data = data[train_num+dev_num:]
    if mini_data:
        #选择前500条样本测试
        train_data = train_data[:500]
        dev_data = dev_data[:100]
        test_data = test_data[:100]
    with open(train_file, 'w', encoding='utf-8') as f:
        for d in train_data:
            f.write(json.dumps(d) + '\n')
    with open(dev_file, 'w', encoding='utf-8') as f:
        for d in dev_data:
            f.write(json.dumps(d)+ '\n')
    with open(test_file, 'w', encoding='utf-8') as f:
        for d in test_data:
            f.write(json.dumps(d)+ '\n')
    print(f"训练集数量{len(train_data)}, 测试集数量{len(test_data)},开发集数量{len(dev_data)}")

if __name__ == '__main__':
    # gen_rel2id(train_dir='/Users/admin/git/Chinese-Literature-NER-RE-Dataset/relation_extraction/Training')
    gen_data(source_dir='/Users/admin/git/Chinese-Literature-NER-RE-Dataset/relation_extraction/Training', des_dir='/Users/admin/git/OpenNRE/benchmark/liter', mini_data=False, truncate=1000)