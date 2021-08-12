# OpenNRE

**在线的Demo网站([http://opennre.thunlp.ai/](http://opennre.thunlp.ai/)). Try it out!**


OpenNRE是一个开源和可扩展的工具包，它提供了一个统一的框架来实现关系抽取模型。这个软件包是为以下群体设计的：

* **关系抽取新手**。我们有手把手的教程和详细的文档，不仅可以让你使用关系抽取工具，还可以帮助你更好的了解这个领域的研究进展。
* **开发者**。我们简单易用的界面和高性能的实现可以使您在实际应用中的部署更加快捷。此外，我们提供了多个预训练的模型，无需任何训练即可投入生产。
* **研究人员**。通过我们的模块化设计，各种任务设置和度量工具，您可以轻松地对自己的模型进行实验，只需稍加修改。我们还提供了多个最常用的基准，用于不同设置的关系抽取。
* **任何需要提交NLP作业来打动教授的人**。我们的软件包拥有最先进的模型，绝对可以帮助你在同学中脱颖而出!


## 什么是关系抽取

关系抽取是一种自然语言处理(NLP)任务，旨在提取实体(如**Bill Gates**和**Microsoft**)之间的关系(如*founder of*)。例如，从句子*Bill Gates founded Microsoft*中，我们可以抽取关系三（**Bill Gates**，*founder of*，**Microsoft**）。

关系抽取是知识图谱自动构建中的一项重要技术。通过使用关系抽取，我们可以积累抽取新的关系事实，扩展知识图谱，作为机器理解人类世界的一种方式，它有很多下游应用，如问答、推荐系统和搜索引擎。

## How to Cite

A good research work is always accompanied by a thorough and faithful reference. If you use or extend our work, please cite the following paper:

```
@inproceedings{han-etal-2019-opennre,
    title = "{O}pen{NRE}: An Open and Extensible Toolkit for Neural Relation Extraction",
    author = "Han, Xu and Gao, Tianyu and Yao, Yuan and Ye, Deming and Liu, Zhiyuan and Sun, Maosong",
    booktitle = "Proceedings of EMNLP-IJCNLP: System Demonstrations",
    year = "2019",
    url = "https://www.aclweb.org/anthology/D19-3029",
    doi = "10.18653/v1/D19-3029",
    pages = "169--174"
}
```

It's our honor to help you better explore relation extraction with our OpenNRE toolkit!

## Papers and Document

If you want to learn more about neural relation extraction, visit another project of ours ([NREPapers](https://github.com/thunlp/NREPapers)).

You can refer to our [document](https://opennre-docs.readthedocs.io/en/latest/) for more details about this project.

## 安装 

### Install as A Python Package

We are now working on deploy OpenNRE as a Python package. Coming soon!

### Using Git Repository

Clone the repository from our github page (don't forget to star us!)

```bash
git clone https://github.com/thunlp/OpenNRE.git
```

If it is too slow, you can try
```
git clone https://github.com/thunlp/OpenNRE.git --depth 1
```

Then install all the requirements:

```
pip install -r requirements.txt
```

**Note**: Please choose appropriate PyTorch version based on your machine (related to your CUDA version). For details, refer to https://pytorch.org/. 

Then install the package with 
```
python setup.py install 
```

If you also want to modify the code, run this:
```
python setup.py develop
```
### 数据集下载
请注意，为了快速部署，我们已经移除了所有数据和预训练文件。你可以通过运行``benchmark``和``pretrain``文件夹中的脚本来手动下载它们。例如，如果你想下载FewRel数据集，你可以运行 "benchmark "和 "pretrain "文件夹中的脚本。

```bash
bash benchmark/download_fewrel.sh
```

## Easy Start

确保你已经按照上面的方法表明安装了OpenNRE。然后导入我们的软件包，并加载预训练好的模型。

```python
>>> import opennre
>>> model = opennre.get_model('wiki80_cnn_softmax')
```

注意，首先下载checkpoint和数据可能需要几分钟。然后使用`infer`进行句子级关系抽取

```python
>>> model.infer({'text': 'He was the son of Máel Dúin mac Máele Fithrich, and grandson of the high king Áed Uaridnach (died 612).', 'h': {'pos': (18, 46)}, 't': {'pos': (78, 91)}})
('father', 0.5108704566955566)
```

得到关系结果和它的置信度分数。
目前，我们有以下几种可用的模型。

* `wiki80_cnn_softmax`: trained on `wiki80` dataset with a CNN encoder.
* `wiki80_bert_softmax`: trained on `wiki80` dataset with a BERT encoder.
* `wiki80_bertentity_softmax`: trained on `wiki80` dataset with a BERT encoder (using entity representation concatenation).
* `tacred_bert_softmax`: trained on `TACRED` dataset with a BERT encoder.
* `tacred_bertentity_softmax`: trained on `TACRED` dataset with a BERT encoder (using entity representation concatenation).

## Training

你可以用OpenNRE在自己的数据上训练自己的模型。在 "example"文件夹中，我们给出了有监督型RE模型和bag-level RE模型的训练代码样本，您可以使用我们提供的数据集或您自己的数据集。

```buildoutcfg
example/
├── train_bag_pcnn_att.py       #使用pcnn，根据论文来看，bert效果较好
├── train_supervised_bert.py    #训练和测试bert模型,可以加选项，使用cls或entity实体的向量表示2种情况
└── train_supervised_cnn.py     #使用cnn

cd OpenNRE;
#确保模型文件下载到了pretrain/bert-base-uncased目录下
python example/train_supervised_bert.py --pretrain_path pretrain/bert-base-uncased --dataset wiki80 --pooler entity --do_train --do_test

#如果没有下载， 那么指定默认名称
python example/train_supervised_bert.py --pretrain_path bert-base-uncased --dataset wiki80 --pooler entity --do_train --do_test

2021-03-15 08:07:32,414 - root - WARNING - Test file ./benchmark/wiki80/wiki80_test.txt does not exist! Use val file instead
2021-03-15 08:07:32,414 - root - INFO - 参数:
2021-03-15 08:07:32,414 - root - INFO -     pretrain_path: bert-base-uncased
2021-03-15 08:07:32,414 - root - INFO -     ckpt: wiki80_bert-base-uncased_entity
2021-03-15 08:07:32,414 - root - INFO -     pooler: entity
2021-03-15 08:07:32,414 - root - INFO -     only_test: False
2021-03-15 08:07:32,414 - root - INFO -     mask_entity: False
2021-03-15 08:07:32,414 - root - INFO -     metric: acc
2021-03-15 08:07:32,414 - root - INFO -     dataset: wiki80
2021-03-15 08:07:32,415 - root - INFO -     train_file: ./benchmark/wiki80/wiki80_train.txt
2021-03-15 08:07:32,415 - root - INFO -     val_file: ./benchmark/wiki80/wiki80_val.txt
2021-03-15 08:07:32,415 - root - INFO -     test_file: ./benchmark/wiki80/wiki80_val.txt
2021-03-15 08:07:32,415 - root - INFO -     rel2id_file: ./benchmark/wiki80/wiki80_rel2id.json
2021-03-15 08:07:32,415 - root - INFO -     batch_size: 16
2021-03-15 08:07:32,415 - root - INFO -     lr: 2e-05
2021-03-15 08:07:32,415 - root - INFO -     max_length: 128
2021-03-15 08:07:32,415 - root - INFO -     max_epoch: 3
2021-03-15 08:07:32,415 - root - INFO - 加载 BERT pre-trained checkpoint.
2021-03-15 08:07:32,630 - filelock - INFO - Lock 139806039272528 acquired on /root/.cache/huggingface/transformers/3c61d016573b14f7f008c02c4e51a366c67ab274726fe2910691e2a761acf43e.637c6035640bacb831febcc2b7f7bee0a96f9b30c2d7e9ef84082d9f252f3170.lock
Downloading: 100% 433/433 [00:00<00:00, 553kB/s]
2021-03-15 08:07:32,845 - filelock - INFO - Lock 139806039272528 released on /root/.cache/huggingface/transformers/3c61d016573b14f7f008c02c4e51a366c67ab274726fe2910691e2a761acf43e.637c6035640bacb831febcc2b7f7bee0a96f9b30c2d7e9ef84082d9f252f3170.lock
2021-03-15 08:07:33,050 - filelock - INFO - Lock 139806178812176 acquired on /root/.cache/huggingface/transformers/a8041bf617d7f94ea26d15e218abd04afc2004805632abc0ed2066aa16d50d04.faf6ea826ae9c5867d12b22257f9877e6b8367890837bd60f7c54a29633f7f2f.lock
Downloading: 100% 440M/440M [00:06<00:00, 65.1MB/s]
2021-03-15 08:07:40,025 - filelock - INFO - Lock 139806178812176 released on /root/.cache/huggingface/transformers/a8041bf617d7f94ea26d15e218abd04afc2004805632abc0ed2066aa16d50d04.faf6ea826ae9c5867d12b22257f9877e6b8367890837bd60f7c54a29633f7f2f.lock
2021-03-15 08:07:42,691 - filelock - INFO - Lock 139806022911376 acquired on /root/.cache/huggingface/transformers/45c3f7a79a80e1cf0a489e5c62b43f173c15db47864303a55d623bb3c96f72a5.d789d64ebfe299b0e416afc4a169632f903f693095b4629a7ea271d5a0cf2c99.lock
Downloading: 100% 232k/232k [00:00<00:00, 921kB/s]
2021-03-15 08:07:43,149 - filelock - INFO - Lock 139806022911376 released on /root/.cache/huggingface/transformers/45c3f7a79a80e1cf0a489e5c62b43f173c15db47864303a55d623bb3c96f72a5.d789d64ebfe299b0e416afc4a169632f903f693095b4629a7ea271d5a0cf2c99.lock
2021-03-15 08:07:46,914 - root - INFO - 加载 RE 数据集 ./benchmark/wiki80/wiki80_train.txt with 50400 行和80 个关系.
2021-03-15 08:07:47,282 - root - INFO - 加载 RE 数据集 ./benchmark/wiki80/wiki80_val.txt with 5600 行和80 个关系.
2021-03-15 08:07:47,882 - root - INFO - 加载 RE 数据集 ./benchmark/wiki80/wiki80_val.txt with 5600 行和80 个关系.
2021-03-15 08:07:47,885 - root - INFO - 检测到GPU可用，使用GPU
2021-03-15 08:07:50,794 - root - INFO - === Epoch 0 train ===
100% 3150/3150 [21:27<00:00,  2.45it/s, acc=0.806, loss=0.788]
2021-03-15 08:29:18,109 - root - INFO - === Epoch 0 val ===
100% 350/350 [00:51<00:00,  6.76it/s, acc=0.858]
2021-03-15 08:30:09,911 - root - INFO - Evaluation result: {'acc': 0.8576785714285714, 'micro_p': 0.8576785714285714, 'micro_r': 0.8576785714285714, 'micro_f1': 0.8576785714285715}.
2021-03-15 08:30:09,911 - root - INFO - Metric micro_f1 current / best: 0.8576785714285715 / 0
2021-03-15 08:30:09,911 - root - INFO - Best ckpt and saved.
2021-03-15 08:30:11,301 - root - INFO - === Epoch 1 train ===
100% 3150/3150 [21:37<00:00,  2.43it/s, acc=0.928, loss=0.245]
2021-03-15 08:51:48,797 - root - INFO - === Epoch 1 val ===
100% 350/350 [00:51<00:00,  6.78it/s, acc=0.867]
2021-03-15 08:52:40,398 - root - INFO - Evaluation result: {'acc': 0.8669642857142857, 'micro_p': 0.8669642857142857, 'micro_r': 0.8669642857142857, 'micro_f1': 0.8669642857142857}.
2021-03-15 08:52:40,398 - root - INFO - Metric micro_f1 current / best: 0.8669642857142857 / 0.8576785714285715
2021-03-15 08:52:40,398 - root - INFO - Best ckpt and saved.
2021-03-15 08:52:41,666 - root - INFO - === Epoch 2 train ===
100% 3150/3150 [21:18<00:00,  2.46it/s, acc=0.957, loss=0.149]
2021-03-15 09:14:00,638 - root - INFO - === Epoch 2 val ===
100% 350/350 [00:51<00:00,  6.77it/s, acc=0.87]
2021-03-15 09:14:52,355 - root - INFO - Evaluation result: {'acc': 0.8703571428571428, 'micro_p': 0.8703571428571428, 'micro_r': 0.8703571428571428, 'micro_f1': 0.8703571428571429}.
2021-03-15 09:14:52,355 - root - INFO - Metric micro_f1 current / best: 0.8703571428571429 / 0.8669642857142857
2021-03-15 09:14:52,355 - root - INFO - Best ckpt and saved.
2021-03-15 09:14:53,672 - root - INFO - Best micro_f1 on val set: 0.870357
100% 350/350 [00:51<00:00,  6.76it/s, acc=0.87]
2021-03-15 09:15:45,670 - root - INFO - Evaluation result: {'acc': 0.8703571428571428, 'micro_p': 0.8703571428571428, 'micro_r': 0.8703571428571428, 'micro_f1': 0.8703571428571429}.
2021-03-15 09:15:45,670 - root - INFO - Test set results:
2021-03-15 09:15:45,670 - root - INFO - Accuracy: 0.8703571428571428
2021-03-15 09:15:45,670 - root - INFO - Micro precision: 0.8703571428571428
2021-03-15 09:15:45,670 - root - INFO - Micro recall: 0.8703571428571428
2021-03-15 09:15:45,670 - root - INFO - Micro F1: 0.8703571428571429

#使用cls的方式, 效果比实体的方式差
python example/train_supervised_bert.py --pretrain_path pretrain/bert-base-uncased --dataset wiki80 --pooler cls --do_train --do_test

2021-03-16 01:11:58,524 - root - INFO - === Epoch 0 train ===
100% 3150/3150 [19:11<00:00,  2.73it/s, acc=0.668, loss=1.44]
2021-03-16 01:31:10,290 - root - INFO - === Epoch 0 val ===
100% 350/350 [00:45<00:00,  7.74it/s, acc=0.798]
2021-03-16 01:31:55,504 - root - INFO - Evaluation result: {'acc': 0.7983928571428571, 'micro_p': 0.7983928571428571, 'micro_r': 0.7983928571428571, 'micro_f1': 0.7983928571428571}.
2021-03-16 01:31:55,504 - root - INFO - Metric micro_f1 current / best: 0.7983928571428571 / 0
2021-03-16 01:31:55,504 - root - INFO - Best ckpt and saved.
2021-03-16 01:31:56,855 - root - INFO - === Epoch 1 train ===
100% 3150/3150 [19:20<00:00,  2.72it/s, acc=0.876, loss=0.446]
2021-03-16 01:51:17,022 - root - INFO - === Epoch 1 val ===
100% 350/350 [00:45<00:00,  7.73it/s, acc=0.838]
2021-03-16 01:52:02,291 - root - INFO - Evaluation result: {'acc': 0.8382142857142857, 'micro_p': 0.8382142857142857, 'micro_r': 0.8382142857142857, 'micro_f1': 0.8382142857142857}.
2021-03-16 01:52:02,291 - root - INFO - Metric micro_f1 current / best: 0.8382142857142857 / 0.7983928571428571
2021-03-16 01:52:02,292 - root - INFO - Best ckpt and saved.
2021-03-16 01:52:03,477 - root - INFO - === Epoch 2 train ===
100% 3150/3150 [19:19<00:00,  2.72it/s, acc=0.923, loss=0.278]
2021-03-16 02:11:22,763 - root - INFO - === Epoch 2 val ===
100% 350/350 [00:45<00:00,  7.75it/s, acc=0.85]
2021-03-16 02:12:07,930 - root - INFO - Evaluation result: {'acc': 0.8498214285714286, 'micro_p': 0.8498214285714286, 'micro_r': 0.8498214285714286, 'micro_f1': 0.8498214285714286}.
2021-03-16 02:12:07,930 - root - INFO - Metric micro_f1 current / best: 0.8498214285714286 / 0.8382142857142857
2021-03-16 02:12:07,930 - root - INFO - Best ckpt and saved.
2021-03-16 02:12:09,198 - root - INFO - Best micro_f1 on val set: 0.849821
100% 350/350 [00:45<00:00,  7.72it/s, acc=0.85]
2021-03-16 02:12:54,762 - root - INFO - Evaluation result: {'acc': 0.8498214285714286, 'micro_p': 0.8498214285714286, 'micro_r': 0.8498214285714286, 'micro_f1': 0.8498214285714286}.
2021-03-16 02:12:54,762 - root - INFO - Test set results:
2021-03-16 02:12:54,763 - root - INFO - Accuracy: 0.8498214285714286
2021-03-16 02:12:54,763 - root - INFO - Micro precision: 0.8498214285714286
2021-03-16 02:12:54,763 - root - INFO - Micro recall: 0.8498214285714286
2021-03-16 02:12:54,763 - root - INFO - Micro F1: 0.8498214285714286
```


# 测试semeval数据集
python example/train_supervised_bert.py --pretrain_path bert-base-uncased --dataset semeval --pooler entity --do_train --do_test
2021-08-12 16:27:15,142 - root - INFO - 加载 RE 数据集 ./benchmark/semeval/semeval_train.txt with 6507 行和19 个关系.
2021-08-12 16:27:15,200 - root - INFO - 加载 RE 数据集 ./benchmark/semeval/semeval_val.txt with 1493 行和19 个关系.
2021-08-12 16:27:15,306 - root - INFO - 加载 RE 数据集 ./benchmark/semeval/semeval_test.txt with 2717 行和19 个关系.
2021-08-12 16:27:15,334 - root - INFO - 检测到GPU可用，使用GPU
2021-08-12 16:27:17,546 - root - INFO - === Epoch 0 train ===
100%|███████████████████████████| 407/407 [00:32<00:00, 12.70it/s, acc=0.574, loss=1.42]
2021-08-12 16:27:49,597 - root - INFO - === Epoch 0 val ===
评估: 100%|██████████████████████████████████| 94/94 [00:02<00:00, 36.45it/s, acc=0.801]
2021-08-12 16:27:52,177 - root - INFO - 评估结果 : {'acc': 0.8010716677829873, 'micro_p': 0.8159695817490494, 'micro_r': 0.8730675345809601, 'micro_f1': 0.8435534591194969}.
2021-08-12 16:27:52,177 - root - INFO - Metric micro_f1 current / best: 0.8435534591194969 / 0
2021-08-12 16:27:52,177 - root - INFO - 获得了更好的metric 0.8435534591194969,保存模型
2021-08-12 16:27:52,611 - root - INFO - === Epoch 1 train ===
100%|███████████████████████████| 407/407 [00:31<00:00, 12.73it/s, acc=0.88, loss=0.401]
2021-08-12 16:28:24,576 - root - INFO - === Epoch 1 val ===
评估: 100%|███████████████████████████████████| 94/94 [00:02<00:00, 36.34it/s, acc=0.85]
2021-08-12 16:28:27,163 - root - INFO - 评估结果 : {'acc': 0.8499665103817816, 'micro_p': 0.8696682464454977, 'micro_r': 0.8958502847843776, 'micro_f1': 0.882565130260521}.
2021-08-12 16:28:27,163 - root - INFO - Metric micro_f1 current / best: 0.882565130260521 / 0.8435534591194969
2021-08-12 16:28:27,164 - root - INFO - 获得了更好的metric 0.882565130260521,保存模型
2021-08-12 16:28:27,901 - root - INFO - === Epoch 2 train ===
100%|██████████████████████████| 407/407 [00:32<00:00, 12.68it/s, acc=0.956, loss=0.177]
2021-08-12 16:29:00,008 - root - INFO - === Epoch 2 val ===
评估: 100%|██████████████████████████████████| 94/94 [00:02<00:00, 36.15it/s, acc=0.848]
2021-08-12 16:29:02,610 - root - INFO - 评估结果 : {'acc': 0.8479571332886805, 'micro_p': 0.8715083798882681, 'micro_r': 0.8885272579332791, 'micro_f1': 0.8799355358581789}.
2021-08-12 16:29:02,610 - root - INFO - Metric micro_f1 current / best: 0.8799355358581789 / 0.882565130260521
2021-08-12 16:29:02,610 - root - INFO - Best micro_f1 on val set: 0.882565
评估: 100%|████████████████████████████████| 170/170 [00:04<00:00, 35.91it/s, acc=0.842]
2021-08-12 16:29:07,473 - root - INFO - 评估结果 : {'acc': 0.8421052631578947, 'micro_p': 0.8738156761412575, 'micro_r': 0.8965974370304906, 'micro_f1': 0.8850599781897494}.
2021-08-12 16:29:07,473 - root - INFO - Test set results:
2021-08-12 16:29:07,473 - root - INFO - Accuracy: 0.8421052631578947
2021-08-12 16:29:07,473 - root - INFO - Micro precision: 0.8738156761412575
2021-08-12 16:29:07,473 - root - INFO - Micro recall: 0.8965974370304906
2021-08-12 16:29:07,473 - root - INFO - Micro F1: 0.8850599781897494

## Google Group

If you want to receive our update news or take part in discussions, please join our [Google Group](https://groups.google.com/forum/#!forum/opennre/join)
