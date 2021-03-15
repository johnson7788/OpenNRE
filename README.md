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
python example/train_supervised_bert.py --pretrain_path pretrain/bert-base-uncased --dataset wiki80
```

## Google Group

If you want to receive our update news or take part in discussions, please join our [Google Group](https://groups.google.com/forum/#!forum/opennre/join)
