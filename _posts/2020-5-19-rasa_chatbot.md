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

# Rasa概述

__Rasa是一个开源的基于机器学习的chatbot开发框架__

其主要分成两大模块：
* Rasa NLU
* Rasa Core

___使用 Rasa NLU + Rasa Core，开发者可以迅速构建自己的chatbot___

本文首先介绍基于任务型对话系统（Task-Bot）的主要概念，然后分析了Rasa的结构组成，介绍开发者如何方便地利用Rasa构建自己的chatbot

# 任务型对话系统（Task-Bot）
自然语言理解（NLU）和对话管理是任务型对话的主要模块。自然语言理解是问答系统、聊天机器人等更高级应用的基石。
### 典型对话系统
* 检索型问答系统（IR-bot）: 主要针对问答系统，提一个问题，给一个答案，不需要参考上下文内容的形式。
* 任务型对话系统（Task-bot）: 针对查询业务，订票之类的任务型对话。
* 闲聊系统（Chitchat-bot）: 像微软小冰，apple Siri等主要陪聊天等。
### Task-Bot
任务型对话系统示意图：
![rasa](/img/rasa1.jpeg)

任务型对话主要包括四部分: 语音识别(ASR)，自然语言理解(NLU)，对话管理(DM)，最后是自然语言生成(NLG)。

下面是一个订餐应用的例子:
![rasa](/img/rasa2.jpeg)

接下来分别来看每个模块具体实现的方式

首先是自然语言理解(NLU)。做自然语言理解首先要有一种表示自然语言含义的形式，一般用传统的三元组方式即：
action, slot , value。action就是意图，slot是需要填充的槽值，value是对应的值。
![rasa](/img/rasa3.jpeg)

具体可以用哪些技术做这些事情呢？下面列出了几个方法。
* 语法分析，可以通过语法规则去分析一句话，得到这句活是疑问句还是肯定句，继而分析出用户意图。相应的也可以通过语法结构中找到对应的槽值。
* 生成模式，主要两个代表性的HMM，CRF, 这样就需要标注数据。
* 分类思想，先对一句话提取特征，再根据有多少个槽值或意图训练多少个分类器，输入一句话分别给不同的分类器，最终得到包含槽值的概率有多大，最终得到这个槽值。
* 深度学习，使用 LSTM+CRF 两种组合的方式进行实体识别，现在也是首选的方法，但有一个问题是深度学习的速度比较慢，一般轻量型的对话系统还是通过语法分析或分类方式或序列标注来做。
![rasa](/img/rasa4.jpeg)

对话状态应该包含持续对话所需要的各种信息。DST的主要作用是记录当前对话状态，作为决策模块的训练数据。
![rasa](/img/rasa5.jpeg)

系统如何做出反馈动作？
![rasa](/img/rasa6.jpeg)
















