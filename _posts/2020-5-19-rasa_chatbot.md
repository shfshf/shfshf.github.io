---
layout:     post
title:      Rasa_chatbot
subtitle:   开发者迅速构建自己的chatbot
date:       2020-5-19
author:     SHF
header-img: img/post-bg-github-cup.jpg
catalog: true
tags:
    - NLP
    - Chatbot
    - DeepLearning
    - Dialogue system
---

# 1.Rasa概述

__Rasa是一个开源的基于机器学习的chatbot开发框架__

其主要分成两大模块：
* Rasa NLU
* Rasa Core

___使用 Rasa NLU + Rasa Core，开发者可以迅速构建自己的chatbot___

本文首先介绍基于任务型对话系统（Task-Bot）的主要概念，然后分析了Rasa的结构组成，介绍开发者如何方便地利用Rasa构建自己的chatbot

# 2.任务型对话系统（Task-Bot）
自然语言理解（NLU）和对话管理是任务型对话的主要模块。自然语言理解是问答系统、聊天机器人等更高级应用的基石。
### 2.1 典型对话系统
* 检索型问答系统（IR-bot）: 主要针对问答系统，提一个问题，给一个答案，不需要参考上下文内容的形式。
* 任务型对话系统（Task-bot）: 针对查询业务，订票之类的任务型对话。
* 闲聊系统（Chitchat-bot）: 像微软小冰，apple Siri等主要陪聊天等。

### 2.2 Task-Bot
任务型对话系统示意图：
![rasa](/img/rasa1.png)

任务型对话主要包括四部分: 语音识别(ASR)，自然语言理解(NLU)，对话管理(DM)，最后是自然语言生成(NLG)。

下面是一个订餐应用的例子:
![rasa](/img/rasa2.png)

接下来分别来看每个模块具体实现的方式

首先是自然语言理解(NLU)。做自然语言理解首先要有一种表示自然语言含义的形式，一般用传统的三元组方式即：
action, slot , value。action就是意图，slot是需要填充的槽值，value是对应的值。
![rasa](/img/rasa3.png)

具体可以用哪些技术做这些事情呢？下面列出了几个方法。
* **语法分析**，可以通过语法规则去分析一句话，得到这句活是疑问句还是肯定句，继而分析出用户意图。相应的也可以通过语法结构中找到对应的槽值。
* **生成模式**，主要两个代表性的HMM，CRF, 这样就需要标注数据。
* **分类思想**，先对一句话提取特征，再根据有多少个槽值或意图训练多少个分类器，输入一句话分别给不同的分类器，最终得到包含槽值的概率有多大，最终得到这个槽值。
* **深度学习**，使用 LSTM+CRF 两种组合的方式进行实体识别，现在也是首选的方法，但有一个问题是深度学习的速度比较慢，一般轻量型的对话系统还是通过语法分析或分类方式或序列标注来做。
![rasa](/img/rasa4.png)

对话状态应该包含持续对话所需要的各种信息。DST的主要作用是记录当前对话状态，作为决策模块的训练数据。
![rasa](/img/rasa5.png)

系统如何做出反馈动作？
![rasa](/img/rasa6.png)

# 3. Rasa 结构

### 3.1 Rasa NLU

Rasa NLU负责提供自然语言理解的工具，包括意图分类和实体抽取。

举例来说，对于输入：
```
帮我找一间市中心的西餐厅
```
Rasa NLU的输出是：
```
{
  "intent": "search_restaurant",
  "entities": {
    "cuisine" : "西餐厅",
    "location" : "市中心"
  }
}
```
其中，*Intent* 代表用户意图。*Entities* 即实体，代表用户输入语句的细节信息。

### 3.2 预定义的pipeline

rasa nlu 支持不同的 Pipeline，其后端实现可支持 spaCy、MITIE、MITIE + sklearn、tensorflow等，其中 spaCy 是官方推荐的。

本文使用的 pipeline 为 MITIE+Jieba+sklearn， rasa nlu 的配置文件为 ivr_chatbot.yml如下：
```
language: "zh"
project: "ivr_nlu"
fixed_model_name: "demo"
path: "models"
pipeline:
- name: "nlp_mitie"
  model: "data/total_word_feature_extractor.dat"  // 加载 mitie 模型
- name: "tokenizer_jieba"  // 使用 jieba 进行分词
- name: "ner_mitie"  // mitie 的命名实体识别
- name: "ner_synonyms"  // 同义词替换
- name: "intent_entity_featurizer_regex" //
- name: "intent_featurizer_mitie"  // 特征提取
- name: "intent_classifier_sklearn"  // sklearn 的意图分类模型
```
每个组件都可以实现Component基类中的多个方法;在管道中，这些不同的方法将按特定的顺序调用。
假设，添加了以下管道到配置："pipeline": ["Component A", "Component B", "Last Component"]。
下图为该管道训练时的调用顺序：![rasa](/img/rasa7.png)


### 3.3 Preparation Work

由于在 pipeline 中使用了 MITIE，所以需要一个训练好的 MITIE 模型。MITIE 模型是非监督训练得到的，类似于 word2vec 中的 word embedding，
需要大量的中文语料，由于训练这个模型对内存要求较高，并且耗时很长，这里直接使用了网友分享的中文 wikipedia 和百度百科语料生成的模型文件 total_word_feature_extractor_chi.dat。

实际应用中，如果做某个特定领域的 NLU 并收集了很多该领域的语料，可以自己去训练 MITIE 模型，也可以用attention，lstm，bert 来预训练词向量。

### 3.4 构建 rasa_nlu 语料

得到 MITIE 词向量模型以后便可以借助其训练 Rasa NLU 模型，这里需要使用标注好的数据来训练 rasa_nlu，标注的数据格式如下：
Rasa 也很贴心的提供了数据标注平台[*rasa-nlu-trainer*](https://rasahq.github.io/rasa-nlu-trainer/) 供用户标注数据。这里我们使用项目里提供的标注好的数据（mobile_nlu_data.json）直接进行训练。
```
# mobile_nlu_data.json
{
        "text": "帮我查一下我十二月消费多少",
        "intent": "request_search",
        "entities": [
          {
            "start": 9,
            "end": 11,
            "value": "消费",
            "entity": "item"
          },
          {
            "start": 6,
            "end": 9,
            "value": "十二月",
            "entity": "time"
          }
        ]
},
.....

```
### 3.5 训练 rasa_nlu 模型



















