# -*- coding: utf-8 -*-
# @Time    : 2021/10/28 15:38
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : cnn_model.py
# @Software: PyCharm

from torch import nn
from torch.nn import Sequential, Conv1d, Linear, Module, Embedding, Dropout, functional
import torchkeras

"""
    TextCNN + Word2Vec

"""


class Config():

    def __init__(self) -> None:
        self.num_embeddings = 1
        self.embedding_dim = 100
        self.dropout = 0.5


class TextCNN(Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        # 参数 :
        # embedding_dim : 单个单词的词向量的大小, 比如 word => 100 维的向量
        self.embedding = Embedding(num_embeddings=config.num_embeddings, embedding_dim=config.embedding_dim)

        # 单个句子长度为 seq_len , 单个单词的向量长度为 100, 句子总数 => [seq_len, 100, sent_len]
        self.dropout = Dropout(config.dropout)
        # 卷积层
        self.conv1d = Conv1d(in_channels=1, out_channels=2, kernel_size=3, stride=1)
        # 全链接层
        self.decoder = Linear(in_features=1, out_features=2)
        # 最大池化层
        self.max_pool = functional.max_pool1d()

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        # 变换维度
        # 原本的维度 (0, 1, 2) => (0, 2, 1) :
        # exp : (2, 3, 4) => (2, 4, 3)
        embedded = embedded.permute(0, 2, 1)
        # 卷积
        out = self.conv1d(embedded)
        out = functional.relu(out)
        # 池化
        out = self.max_pool(out)
        # 全链接层
        out = self.decoder(out)
        return out
