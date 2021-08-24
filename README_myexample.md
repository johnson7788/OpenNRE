# 已经集成到多任务模型，现有代码暂时不用

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
2021-08-16 15:08:52,443 - root - INFO - 参数:
2021-08-16 15:08:52,443 - root - INFO -     pretrain_path: pretrain/bert_model
2021-08-16 15:08:52,443 - root - INFO -     ckpt: brand_pretrain/bert_model_entity
2021-08-16 15:08:52,444 - root - INFO -     pooler: entity
2021-08-16 15:08:52,444 - root - INFO -     do_train: True
2021-08-16 15:08:52,444 - root - INFO -     do_test: True
2021-08-16 15:08:52,444 - root - INFO -     mask_entity: False
2021-08-16 15:08:52,444 - root - INFO -     metric: micro_f1
2021-08-16 15:08:52,444 - root - INFO -     dataset: brand
2021-08-16 15:08:52,444 - root - INFO -     train_file: ./benchmark/brand/brand_train.txt
2021-08-16 15:08:52,444 - root - INFO -     val_file: ./benchmark/brand/brand_val.txt
2021-08-16 15:08:52,444 - root - INFO -     test_file: ./benchmark/brand/brand_test.txt
2021-08-16 15:08:52,444 - root - INFO -     rel2id_file: ./benchmark/brand/brand_rel2id.json
2021-08-16 15:08:52,444 - root - INFO -     batch_size: 32
2021-08-16 15:08:52,444 - root - INFO -     lr: 2e-05
2021-08-16 15:08:52,444 - root - INFO -     max_length: 128
2021-08-16 15:08:52,444 - root - INFO -     max_epoch: 10
2021-08-16 15:08:52,444 - root - INFO - 加载预训练的 BERT pre-trained checkpoint: pretrain/bert_model
Some weights of the model checkpoint at pretrain/bert_model were not used when initializing BertModel: ['cls.predictions.decoder.bias', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.weight', 'cls.predictions.bias', 'cls.predictions.decoder.weight']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
2021-08-16 15:08:54,180 - root - INFO - 加载 RE 数据集 ./benchmark/brand/brand_train.txt with 10598 行和2 个关系.
2021-08-16 15:08:54,298 - root - INFO - 加载 RE 数据集 ./benchmark/brand/brand_val.txt with 1326 行和2 个关系.
2021-08-16 15:08:54,416 - root - INFO - 加载 RE 数据集 ./benchmark/brand/brand_test.txt with 1324 行和2 个关系.
2021-08-16 15:08:54,447 - root - INFO - 检测到GPU可用，使用GPU
2021-08-16 15:08:56,674 - root - INFO - === Epoch 0 train ===
100%|██████████| 332/332 [01:07<00:00,  4.95it/s, acc=0.75, loss=0.528] 
2021-08-16 15:10:03,678 - root - INFO - === Epoch 0 val ===
评估: 100%|██████████| 42/42 [00:04<00:00,  8.71it/s, acc=0.759]
2021-08-16 15:10:08,500 - root - INFO - 评估结果 : {'acc': 0.7594268476621417, 'micro_p': 0.7594268476621417, 'micro_r': 0.7594268476621417, 'micro_f1': 0.7594268476621419}.
2021-08-16 15:10:08,500 - root - INFO - Metric micro_f1 current / best: 0.7594268476621419 / 0
2021-08-16 15:10:08,500 - root - INFO - 获得了更好的metric 0.7594268476621419,保存模型
2021-08-16 15:10:09,188 - root - INFO - === Epoch 1 train ===
100%|██████████| 332/332 [01:06<00:00,  4.98it/s, acc=0.809, loss=0.429]
2021-08-16 15:11:15,790 - root - INFO - === Epoch 1 val ===
评估: 100%|██████████| 42/42 [00:04<00:00,  8.73it/s, acc=0.81] 
2021-08-16 15:11:20,604 - root - INFO - 评估结果 : {'acc': 0.8099547511312217, 'micro_p': 0.8099547511312217, 'micro_r': 0.8099547511312217, 'micro_f1': 0.8099547511312217}.
2021-08-16 15:11:20,604 - root - INFO - Metric micro_f1 current / best: 0.8099547511312217 / 0.7594268476621419
2021-08-16 15:11:20,604 - root - INFO - 获得了更好的metric 0.8099547511312217,保存模型
2021-08-16 15:11:21,302 - root - INFO - === Epoch 2 train ===
100%|██████████| 332/332 [01:06<00:00,  4.97it/s, acc=0.84, loss=0.357] 
2021-08-16 15:12:28,068 - root - INFO - === Epoch 2 val ===
评估: 100%|██████████| 42/42 [00:04<00:00,  8.71it/s, acc=0.825]
2021-08-16 15:12:32,894 - root - INFO - 评估结果 : {'acc': 0.8250377073906485, 'micro_p': 0.8250377073906485, 'micro_r': 0.8250377073906485, 'micro_f1': 0.8250377073906485}.
2021-08-16 15:12:32,894 - root - INFO - Metric micro_f1 current / best: 0.8250377073906485 / 0.8099547511312217
2021-08-16 15:12:32,894 - root - INFO - 获得了更好的metric 0.8250377073906485,保存模型
2021-08-16 15:12:33,554 - root - INFO - === Epoch 3 train ===
100%|██████████| 332/332 [01:06<00:00,  4.96it/s, acc=0.857, loss=0.318]
2021-08-16 15:13:40,473 - root - INFO - === Epoch 3 val ===
评估: 100%|██████████| 42/42 [00:04<00:00,  8.69it/s, acc=0.807]
2021-08-16 15:13:45,309 - root - INFO - 评估结果 : {'acc': 0.8069381598793364, 'micro_p': 0.8069381598793364, 'micro_r': 0.8069381598793364, 'micro_f1': 0.8069381598793365}.
2021-08-16 15:13:45,309 - root - INFO - Metric micro_f1 current / best: 0.8069381598793365 / 0.8250377073906485
2021-08-16 15:13:45,310 - root - INFO - === Epoch 4 train ===
100%|██████████| 332/332 [01:07<00:00,  4.94it/s, acc=0.865, loss=0.296]
2021-08-16 15:14:52,566 - root - INFO - === Epoch 4 val ===
评估: 100%|██████████| 42/42 [00:04<00:00,  8.69it/s, acc=0.821]
2021-08-16 15:14:57,398 - root - INFO - 评估结果 : {'acc': 0.8205128205128205, 'micro_p': 0.8205128205128205, 'micro_r': 0.8205128205128205, 'micro_f1': 0.8205128205128205}.
2021-08-16 15:14:57,398 - root - INFO - Metric micro_f1 current / best: 0.8205128205128205 / 0.8250377073906485
2021-08-16 15:14:57,399 - root - INFO - === Epoch 5 train ===
100%|██████████| 332/332 [01:07<00:00,  4.95it/s, acc=0.872, loss=0.28] 
2021-08-16 15:16:04,527 - root - INFO - === Epoch 5 val ===
评估: 100%|██████████| 42/42 [00:04<00:00,  8.70it/s, acc=0.825]
2021-08-16 15:16:09,357 - root - INFO - 评估结果 : {'acc': 0.8250377073906485, 'micro_p': 0.8250377073906485, 'micro_r': 0.8250377073906485, 'micro_f1': 0.8250377073906485}.
2021-08-16 15:16:09,357 - root - INFO - Metric micro_f1 current / best: 0.8250377073906485 / 0.8250377073906485
2021-08-16 15:16:09,358 - root - INFO - === Epoch 6 train ===
100%|██████████| 332/332 [01:07<00:00,  4.93it/s, acc=0.875, loss=0.264]
2021-08-16 15:17:16,684 - root - INFO - === Epoch 6 val ===
评估: 100%|██████████| 42/42 [00:04<00:00,  8.69it/s, acc=0.811]
2021-08-16 15:17:21,516 - root - INFO - 评估结果 : {'acc': 0.8107088989441931, 'micro_p': 0.8107088989441931, 'micro_r': 0.8107088989441931, 'micro_f1': 0.8107088989441931}.
2021-08-16 15:17:21,517 - root - INFO - Metric micro_f1 current / best: 0.8107088989441931 / 0.8250377073906485
2021-08-16 15:17:21,517 - root - INFO - === Epoch 7 train ===
100%|██████████| 332/332 [01:07<00:00,  4.93it/s, acc=0.878, loss=0.253]
2021-08-16 15:18:28,924 - root - INFO - === Epoch 7 val ===
评估: 100%|██████████| 42/42 [00:04<00:00,  8.70it/s, acc=0.814]
2021-08-16 15:18:33,751 - root - INFO - 评估结果 : {'acc': 0.8144796380090498, 'micro_p': 0.8144796380090498, 'micro_r': 0.8144796380090498, 'micro_f1': 0.8144796380090498}.
2021-08-16 15:18:33,752 - root - INFO - Metric micro_f1 current / best: 0.8144796380090498 / 0.8250377073906485
2021-08-16 15:18:33,752 - root - INFO - === Epoch 8 train ===
100%|██████████| 332/332 [01:07<00:00,  4.93it/s, acc=0.883, loss=0.242]
2021-08-16 15:19:41,049 - root - INFO - === Epoch 8 val ===
评估: 100%|██████████| 42/42 [00:04<00:00,  8.70it/s, acc=0.819]
2021-08-16 15:19:45,879 - root - INFO - 评估结果 : {'acc': 0.8190045248868778, 'micro_p': 0.8190045248868778, 'micro_r': 0.8190045248868778, 'micro_f1': 0.8190045248868778}.
2021-08-16 15:19:45,879 - root - INFO - Metric micro_f1 current / best: 0.8190045248868778 / 0.8250377073906485
2021-08-16 15:19:45,879 - root - INFO - === Epoch 9 train ===
100%|██████████| 332/332 [01:07<00:00,  4.93it/s, acc=0.888, loss=0.236]
2021-08-16 15:20:53,230 - root - INFO - === Epoch 9 val ===
评估: 100%|██████████| 42/42 [00:04<00:00,  8.68it/s, acc=0.808]
2021-08-16 15:20:58,069 - root - INFO - 评估结果 : {'acc': 0.808446455505279, 'micro_p': 0.808446455505279, 'micro_r': 0.808446455505279, 'micro_f1': 0.808446455505279}.
2021-08-16 15:20:58,069 - root - INFO - Metric micro_f1 current / best: 0.808446455505279 / 0.8250377073906485
2021-08-16 15:20:58,069 - root - INFO - Best micro_f1 on val set: 0.825038
评估: 100%|██████████| 42/42 [00:04<00:00,  8.66it/s, acc=0.826]
2021-08-16 15:21:03,038 - root - INFO - 评估结果 : {'acc': 0.8262839879154078, 'micro_p': 0.8262839879154078, 'micro_r': 0.8262839879154078, 'micro_f1': 0.8262839879154078}.
2021-08-16 15:21:03,038 - root - INFO - Test set results:
2021-08-16 15:21:03,038 - root - INFO - Accuracy: 0.8262839879154078
2021-08-16 15:21:03,038 - root - INFO - Micro precision: 0.8262839879154078
2021-08-16 15:21:03,038 - root - INFO - Micro recall: 0.8262839879154078
2021-08-16 15:21:03,038 - root - INFO - Micro F1: 0.8262839879154078
运行成功! Step3: 训练并测试BERT模型
  

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
  
  
# 模型处理cls类型的训练时的方法
```
sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
ent0 = ['[unused0]'] + ent0 + ['[unused1]']
ent1 = ['[unused2]'] + ent1 + ['[unused3]']
re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
```
