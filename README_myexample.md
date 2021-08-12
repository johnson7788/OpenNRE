# 生成中文数据，数据格式
## 例如benchmark/liter
标签文件
```
{"否": 0, "是": 1}
```
具体数据
```
{
    "text":"出门回来、一定要彻底卸妆
lirosa水霜、冻膜、卸妆啫喱还有香奈儿山茶花洗面奶都是无限回购。",
    "h":{
        "name":"lirosa水霜",
        "id":"VKZH9J5DW8",
        "pos":[
            13,
            21
        ]
    },
    "t":{
        "name":"啫喱",
        "id":"U7G1VDPYTG",
        "pos":[
            27,
            29
        ]
    },
    "relation":"否"
}
```

# 模型是从huggface下载好的
#训练模型, 使用macbert模型
python train_supervised_bert.py --pretrain_path pretrain/mac_bert_model --dataset brand --pooler entity --do_train --do_test --batch_size 32 --max_length 256 --max_epoch 10

#使用中文bert模型
## 实体形式
共收集到总的数据条目: 13248, 跳过的空的数据: 153, 非空reuslt的条数3347, 标签为空的数据的条数2，标签的个数统计为Counter({'否': 9051, '是': 4197})
训练集数量10598, 测试集数量1326,开发集数量1324
python train_supervised_bert.py --pretrain_path pretrain/bert_model --dataset brand --pooler entity --do_train --do_test --batch_size 32 --max_length 256 --max_epoch 10
2021-08-12 18:07:58,316 - root - INFO - 参数:
2021-08-12 18:07:58,316 - root - INFO -     pretrain_path: pretrain/bert_model
2021-08-12 18:07:58,316 - root - INFO -     ckpt: brand_pretrain/bert_model_entity
2021-08-12 18:07:58,316 - root - INFO -     pooler: entity
2021-08-12 18:07:58,316 - root - INFO -     do_train: True
2021-08-12 18:07:58,316 - root - INFO -     do_test: True
2021-08-12 18:07:58,316 - root - INFO -     mask_entity: False
2021-08-12 18:07:58,316 - root - INFO -     metric: micro_f1
2021-08-12 18:07:58,316 - root - INFO -     dataset: brand
2021-08-12 18:07:58,316 - root - INFO -     train_file: ./benchmark/brand/brand_train.txt
2021-08-12 18:07:58,316 - root - INFO -     val_file: ./benchmark/brand/brand_val.txt
2021-08-12 18:07:58,316 - root - INFO -     test_file: ./benchmark/brand/brand_test.txt
2021-08-12 18:07:58,316 - root - INFO -     rel2id_file: ./benchmark/brand/brand_rel2id.json
2021-08-12 18:07:58,316 - root - INFO -     batch_size: 32
2021-08-12 18:07:58,316 - root - INFO -     lr: 2e-05
2021-08-12 18:07:58,316 - root - INFO -     max_length: 128
2021-08-12 18:07:58,316 - root - INFO -     max_epoch: 10
2021-08-12 18:07:58,317 - root - INFO - 加载预训练的 BERT pre-trained checkpoint: pretrain/bert_model
Some weights of the model checkpoint at pretrain/bert_model were not used when initializing BertModel: ['cls.predictions.decoder.bias', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.decoder.weight']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
2021-08-12 18:08:00,022 - root - INFO - 加载 RE 数据集 ./benchmark/brand/brand_train.txt with 10598 行和2 个关系.
2021-08-12 18:08:00,140 - root - INFO - 加载 RE 数据集 ./benchmark/brand/brand_val.txt with 1326 行和2 个关系.
2021-08-12 18:08:00,257 - root - INFO - 加载 RE 数据集 ./benchmark/brand/brand_test.txt with 1324 行和2 个关系.
2021-08-12 18:08:00,286 - root - INFO - 检测到GPU可用，使用GPU
2021-08-12 18:08:02,491 - root - INFO - === Epoch 0 train ===
100%|██████████| 332/332 [01:08<00:00,  4.86it/s, acc=0.745, loss=0.537]
2021-08-12 18:09:10,768 - root - INFO - === Epoch 0 val ===
评估: 100%|██████████| 42/42 [00:05<00:00,  8.38it/s, acc=0.801]
2021-08-12 18:09:15,782 - root - INFO - 评估结果 : {'acc': 0.8009049773755657, 'micro_p': 0.8009049773755657, 'micro_r': 0.8009049773755657, 'micro_f1': 0.8009049773755657}.
  

## CLS形式
python train_supervised_bert.py --pretrain_path pretrain/bert_model --dataset brand --pooler cls --do_train --do_test --batch_size 32 --max_length 256 --max_epoch 10
共收集到总的数据条目: 13248, 跳过的空的数据: 153, 非空reuslt的条数3347, 标签为空的数据的条数2，标签的个数统计为Counter({'否': 9051, '是': 4197})
训练集数量10598, 测试集数量1326,开发集数量1324
2021-08-12 17:54:41,666 - root - INFO - 参数:
2021-08-12 17:54:41,666 - root - INFO -     pretrain_path: pretrain/bert_model
2021-08-12 17:54:41,666 - root - INFO -     ckpt: brand_pretrain/bert_model_cls
2021-08-12 17:54:41,666 - root - INFO -     pooler: cls
2021-08-12 17:54:41,666 - root - INFO -     do_train: True
2021-08-12 17:54:41,666 - root - INFO -     do_test: True
2021-08-12 17:54:41,666 - root - INFO -     mask_entity: False
2021-08-12 17:54:41,666 - root - INFO -     metric: micro_f1
2021-08-12 17:54:41,666 - root - INFO -     dataset: brand
2021-08-12 17:54:41,666 - root - INFO -     train_file: ./benchmark/brand/brand_train.txt
2021-08-12 17:54:41,666 - root - INFO -     val_file: ./benchmark/brand/brand_val.txt
2021-08-12 17:54:41,666 - root - INFO -     test_file: ./benchmark/brand/brand_test.txt
2021-08-12 17:54:41,666 - root - INFO -     rel2id_file: ./benchmark/brand/brand_rel2id.json
2021-08-12 17:54:41,666 - root - INFO -     batch_size: 32
2021-08-12 17:54:41,666 - root - INFO -     lr: 2e-05
2021-08-12 17:54:41,666 - root - INFO -     max_length: 128
2021-08-12 17:54:41,666 - root - INFO -     max_epoch: 10
2021-08-12 17:54:41,666 - root - INFO - 加载预训练的 BERT pre-trained checkpoint: pretrain/bert_model
Some weights of the model checkpoint at pretrain/bert_model were not used when initializing BertModel: ['cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight', 'cls.predictions.transform.dense.weight', 'cls.predictions.decoder.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.bias']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
2021-08-12 17:54:43,538 - root - INFO - 加载 RE 数据集 ./benchmark/brand/brand_train.txt with 10598 行和2 个关系.
2021-08-12 17:54:43,656 - root - INFO - 加载 RE 数据集 ./benchmark/brand/brand_val.txt with 1326 行和2 个关系.
2021-08-12 17:54:43,772 - root - INFO - 加载 RE 数据集 ./benchmark/brand/brand_test.txt with 1324 行和2 个关系.
2021-08-12 17:54:43,802 - root - INFO - 检测到GPU可用，使用GPU
2021-08-12 17:54:45,984 - root - INFO - === Epoch 0 train ===
100%|██████████| 332/332 [01:07<00:00,  4.94it/s, acc=0.72, loss=0.581] 
2021-08-12 17:55:53,174 - root - INFO - === Epoch 0 val ===
评估: 100%|██████████| 42/42 [00:04<00:00,  8.74it/s, acc=0.788]
2021-08-12 17:55:57,981 - root - INFO - 评估结果 : {'acc': 0.7880844645550528, 'micro_p': 0.7880844645550528, 'micro_r': 0.7880844645550528, 'micro_f1': 0.7880844645550528}.
2021-08-12 17:55:57,981 - root - INFO - Metric micro_f1 current / best: 0.7880844645550528 / 0
2021-08-12 17:55:57,981 - root - INFO - 获得了更好的metric 0.7880844645550528,保存模型
2021-08-12 17:55:58,365 - root - INFO - === Epoch 1 train ===
 86%|████████▋ | 287/332 [00:58<00:09,  4.81it/s, acc=0.791, loss=0.488]
  2021-08-12 17:58:20,257 - root - INFO - === Epoch 2 val ===
评估: 100%|██████████| 42/42 [00:04<00:00,  8.51it/s, acc=0.818]
2021-08-12 17:58:25,196 - root - INFO - 评估结果 : {'acc': 0.8182503770739065, 'micro_p': 0.8182503770739065, 'micro_r': 0.8182503770739065, 'micro_f1': 0.8182503770739065}.
2021-08-12 17:58:25,196 - root - INFO - Metric micro_f1 current / best: 0.8182503770739065 / 0.8107088989441931
2021-08-12 17:58:25,196 - root - INFO - 获得了更好的metric 0.8182503770739065,保存模型
2021-08-12 17:58:25,885 - root - INFO - === Epoch 3 train ===
100%|██████████| 332/332 [01:08<00:00,  4.82it/s, acc=0.841, loss=0.371]
2021-08-12 17:59:34,769 - root - INFO - === Epoch 3 val ===
评估: 100%|██████████| 42/42 [00:05<00:00,  8.39it/s, acc=0.802]
2021-08-12 17:59:39,776 - root - INFO - 评估结果 : {'acc': 0.801659125188537, 'micro_p': 0.801659125188537, 'micro_r': 0.801659125188537, 'micro_f1': 0.8016591251885369}.
2021-08-12 17:59:39,776 - root - INFO - Metric micro_f1 current / best: 0.8016591251885369 / 0.8182503770739065
2021-08-12 17:59:39,777 - root - INFO - === Epoch 4 train ===
100%|██████████| 332/332 [01:09<00:00,  4.81it/s, acc=0.852, loss=0.34] 
2021-08-12 18:00:48,805 - root - INFO - === Epoch 4 val ===
评估: 100%|██████████| 42/42 [00:05<00:00,  7.71it/s, acc=0.822]
2021-08-12 18:00:54,255 - root - INFO - 评估结果 : {'acc': 0.8220211161387632, 'micro_p': 0.8220211161387632, 'micro_r': 0.8220211161387632, 'micro_f1': 0.8220211161387632}.
2021-08-12 18:00:54,255 - root - INFO - Metric micro_f1 current / best: 0.8220211161387632 / 0.8182503770739065
2021-08-12 18:00:54,256 - root - INFO - 获得了更好的metric 0.8220211161387632,保存模型
2021-08-12 18:00:54,986 - root - INFO - === Epoch 5 train ===
100%|██████████| 332/332 [01:11<00:00,  4.62it/s, acc=0.858, loss=0.311]
2021-08-12 18:02:06,912 - root - INFO - === Epoch 5 val ===
评估: 100%|██████████| 42/42 [00:04<00:00,  8.71it/s, acc=0.834]
2021-08-12 18:02:11,735 - root - INFO - 评估结果 : {'acc': 0.8340874811463047, 'micro_p': 0.8340874811463047, 'micro_r': 0.8340874811463047, 'micro_f1': 0.8340874811463046}.
2021-08-12 18:02:11,735 - root - INFO - Metric micro_f1 current / best: 0.8340874811463046 / 0.8220211161387632
2021-08-12 18:02:11,735 - root - INFO - 获得了更好的metric 0.8340874811463046,保存模型
2021-08-12 18:02:12,439 - root - INFO - === Epoch 6 train ===
100%|██████████| 332/332 [01:09<00:00,  4.78it/s, acc=0.863, loss=0.297]
2021-08-12 18:03:21,880 - root - INFO - === Epoch 6 val ===
评估: 100%|██████████| 42/42 [00:05<00:00,  7.94it/s, acc=0.83] 
2021-08-12 18:03:27,174 - root - INFO - 评估结果 : {'acc': 0.8295625942684767, 'micro_p': 0.8295625942684767, 'micro_r': 0.8295625942684767, 'micro_f1': 0.8295625942684767}.
2021-08-12 18:03:27,174 - root - INFO - Metric micro_f1 current / best: 0.8295625942684767 / 0.8340874811463046
2021-08-12 18:03:27,174 - root - INFO - === Epoch 7 train ===
100%|██████████| 332/332 [01:13<00:00,  4.51it/s, acc=0.869, loss=0.283]
2021-08-12 18:04:40,825 - root - INFO - === Epoch 7 val ===
评估: 100%|██████████| 42/42 [00:04<00:00,  8.56it/s, acc=0.818]
2021-08-12 18:04:45,732 - root - INFO - 评估结果 : {'acc': 0.8182503770739065, 'micro_p': 0.8182503770739065, 'micro_r': 0.8182503770739065, 'micro_f1': 0.8182503770739065}.
2021-08-12 18:04:45,732 - root - INFO - Metric micro_f1 current / best: 0.8182503770739065 / 0.8340874811463046
2021-08-12 18:04:45,733 - root - INFO - === Epoch 8 train ===
100%|██████████| 332/332 [01:09<00:00,  4.79it/s, acc=0.873, loss=0.264]
2021-08-12 18:05:54,999 - root - INFO - === Epoch 8 val ===
评估: 100%|██████████| 42/42 [00:05<00:00,  8.35it/s, acc=0.824]
2021-08-12 18:06:00,030 - root - INFO - 评估结果 : {'acc': 0.8242835595776772, 'micro_p': 0.8242835595776772, 'micro_r': 0.8242835595776772, 'micro_f1': 0.8242835595776772}.
2021-08-12 18:06:00,030 - root - INFO - Metric micro_f1 current / best: 0.8242835595776772 / 0.8340874811463046
2021-08-12 18:06:00,031 - root - INFO - === Epoch 9 train ===
100%|██████████| 332/332 [01:09<00:00,  4.78it/s, acc=0.878, loss=0.256]
2021-08-12 18:07:09,461 - root - INFO - === Epoch 9 val ===
评估: 100%|██████████| 42/42 [00:05<00:00,  7.84it/s, acc=0.82] 
2021-08-12 18:07:14,817 - root - INFO - 评估结果 : {'acc': 0.8197586726998491, 'micro_p': 0.8197586726998491, 'micro_r': 0.8197586726998491, 'micro_f1': 0.8197586726998491}.
2021-08-12 18:07:14,817 - root - INFO - Metric micro_f1 current / best: 0.8197586726998491 / 0.8340874811463046
2021-08-12 18:07:14,817 - root - INFO - Best micro_f1 on val set: 0.834087
评估: 100%|██████████| 42/42 [00:05<00:00,  7.61it/s, acc=0.825]
2021-08-12 18:07:20,456 - root - INFO - 评估结果 : {'acc': 0.824773413897281, 'micro_p': 0.824773413897281, 'micro_r': 0.824773413897281, 'micro_f1': 0.824773413897281}.
2021-08-12 18:07:20,456 - root - INFO - Test set results:
2021-08-12 18:07:20,456 - root - INFO - Accuracy: 0.824773413897281
2021-08-12 18:07:20,456 - root - INFO - Micro precision: 0.824773413897281
2021-08-12 18:07:20,456 - root - INFO - Micro recall: 0.824773413897281
2021-08-12 18:07:20,456 - root - INFO - Micro F1: 0.824773413897281
运行成功! Step3: 训练并测试BERT模型  