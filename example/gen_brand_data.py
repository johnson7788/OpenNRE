#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/3/19 10:58 上午
# @File  : gen_chinese_data.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 根据label-studio标注的数据, 品牌和需求的关系判断
"""
{
  "id": 25,
  "data": {
    "brand": "修丽可age面霜",
    "channel": "redbook",
    "requirement": "保湿,抗老",
    "text": "我和大家一样都是普通上班族，每天面对屏幕的上班族的姐妹们皮肤总是难免的暗黄，搞得好像一下就老了好几岁的样子。工作 车子 房子各种的压力，每天的保养也不是总有时间和精力去做的。\n直到同事给我推荐了这款修丽可age面霜，每天晚上涂抹一点，半个月就感觉自己的肌肤不像平时那么松弛干燥了，效 果可以说是很不错了。\n正直秋冬，天干物燥，作为一款面霜，它的保湿效果还是很明 显的，细小的皱 纹也能有效的抚平，感觉用了之后整个人会明显觉得更 显年 轻，肤质回到了十八九的样子 甚至会恍然大悟原来之前脸干不单单是因为天气，还因为皮肤老化！因为现在整个底子好了，每天化妆都很服帖，不会再有各种卡粉啊暗沉之类的困扰了 \n真心推荐像我一样的上班族入手这款面霜，毕竟抗老真的很重要，钱要花在刀刃上嘛 "
  },
  "completions": [
    {
      "created_at": 1627890313,
      "id": 25001,
      "lead_time": 4.6,
      "result": [
        {
          "from_name": "label",
          "id": "BJPV4AQ0ML",
          "to_name": "text",
          "type": "labels",
          "value": {
            "end": 107,
            "labels": [
              "品牌"
            ],
            "start": 99,
            "text": "修丽可age面霜"
          }
        },
        {
          "from_name": "label",
          "id": "9GU824R5UZ",
          "to_name": "text",
          "type": "labels",
          "value": {
            "end": 174,
            "labels": [
              "需求"
            ],
            "start": 172,
            "text": "保湿"
          }
        },
        {
          "from_name": "label",
          "id": "1G3PAO1G0C",
          "to_name": "text",
          "type": "labels",
          "value": {
            "end": 323,
            "labels": [
              "需求"
            ],
            "start": 321,
            "text": "抗老"
          }
        },
        {
          "direction": "right",
          "from_id": "BJPV4AQ0ML",
          "labels": [
            "是"
          ],
          "to_id": "9GU824R5UZ",
          "type": "relation"
        }
      ]
    }
  ],
  "predictions": []
}
"""

import os
import json
import re
import random

def gen_rel2id(destination='/Users/admin/git/OpenNRE/benchmark/brand/brand_rel2id.json'):
    """
    直接输出关系到id的映射
    :param destination: 输出的目标json文件
    :return:
    """
    rel2id = {"否": 0, "是": 1}
    with open(destination, 'w', encoding='utf-8') as f:
        json.dump(rel2id, f)

def gen_data(source_dir, des_dir):
    """
    根据原始目录生成目标训练或测试等文件
    :param source_dir: eg: 标注的原始数据目录
    :param des_dir:  eg: /Users/admin/git/OpenNRE/benchmark/brand
    :return:
    """
    #保存处理好的数据
    data = []
    # 计数，空的result的数据的个数
    empty_result_num = 0
    #标签是空的数据条数
    empty_labels_num = 0
    #result不是空的数据的条数
    result_num = 0
    files = os.listdir(source_dir)
    # 过滤出标注的文件
    json_files = [f for f in files if f.endswith('.json')]
    for jfile in json_files:
        jfile_path = os.path.join(source_dir, jfile)
        with open(jfile_path, 'r') as f:
            json_data = json.load(f)
        for d in json_data:
            # 包含brand， channel，requirement，和text
            data_content = d['data']
            completions = d['completions']
            # 只选取第一个标注的数据
            result = completions[0]['result']
            if not result:
                # result为空的，过滤掉
                empty_result_num += 1
                continue
            else:
                result_num += 1
            #解析result，标注数据，包含2种，一个是关键字是品牌或需求，另一个是品牌和需求的关系
            brand_requirements = [r for r in result if r.get('from_name')]
            # 变成id和其它属性的字典
            brand_requirements_id_dict = {br['id']:br['value'] for br in brand_requirements}
            relations = [r for r in result if r.get('direction')]
            for rel in relations:
                # 关系, 是或否
                # 如果labels为空，也跳过
                if not rel['labels']:
                    empty_labels_num += 1
                    continue
                relation = rel['labels'][0]
                # 每个关系生成一条数据
                text = data_content['text']
                # 头部实体的名字
                h_id = rel['from_id']
                # 获取头部id对应的名称和位置
                h_value = brand_requirements_id_dict[h_id]
                h_start = h_value['start']
                h_end = h_value['end']
                h_name = h_value['text']
                # 尾部实体的id
                t_id = rel['to_id']
                t_value = brand_requirements_id_dict[t_id]
                t_start = t_value['start']
                t_end = t_value['end']
                t_name = t_value['text']
                #校验数据
                assert relation in ["是","否"]
                one_data = {
                    'text': text,
                    'h': {
                        'name': h_name,
                        'id': h_id,
                        'pos': [h_start, h_end]
                    },
                    't': {
                        'name': t_name,
                        'id': t_id,
                        'pos': [t_start, t_end]
                    },
                    'relation': relation
                }
            data.append(one_data)
    print(f"共收集到总的数据条目: {len(data)}, 跳过的空的数据: {empty_result_num}, 非空reuslt的条数{result_num}, 标签为空的数据的条数{empty_labels_num}")
    train_file = os.path.join(des_dir, 'brand_train.txt')
    dev_file = os.path.join(des_dir, 'brand_test.txt')
    test_file = os.path.join(des_dir, 'brand_val.txt')
    random.seed(6)
    random.shuffle(data)
    train_num = int(len(data) * 0.8)
    dev_num = int(len(data) * 0.1)
    train_data = data[:train_num]
    dev_data = data[train_num:train_num+dev_num]
    test_data = data[train_num+dev_num:]
    with open(train_file, 'w', encoding='utf-8') as f:
        for d in train_data:
            f.write(json.dumps(d,ensure_ascii=False) + '\n')
    with open(dev_file, 'w', encoding='utf-8') as f:
        for d in dev_data:
            f.write(json.dumps(d,ensure_ascii=False)+ '\n')
    with open(test_file, 'w', encoding='utf-8') as f:
        for d in test_data:
            f.write(json.dumps(d,ensure_ascii=False)+ '\n')
    print(f"训练集数量{len(train_data)}, 测试集数量{len(test_data)},开发集数量{len(dev_data)}")

if __name__ == '__main__':
    # gen_rel2id()
    gen_data(source_dir='/opt/lavector/relation/', des_dir='/Users/admin/git/OpenNRE/benchmark/brand/')