# -*- coding: utf-8 -*-
# @Time    : 2021/11/1 21:20
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : embedding_func.py
# @Software: PyCharm
import torch
from torch import nn


embedding = nn.Embedding(5, 3) # 定义一个单词表为5, 维度为3的词向量表

# 打印随机初始化的词向量权重
print(embedding.weight)
print(embedding.weight.size())


# 传入对应的词索引
# {i: 0, love: 1, you: 2}

word2idx = [
    [0, 1, 2],
    [0, 2, 1],
    [0, 0, 1],
]

word2idx = torch.LongTensor(word2idx)

vector = embedding(word2idx)

print(vector)