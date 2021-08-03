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
python train_supervised_bert.py --pretrain_path pretrain/bert_model --dataset brand --pooler entity --do_train --do_test --batch_size 32 --max_length 256 --max_epoch 10