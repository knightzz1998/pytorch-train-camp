# -*- coding: utf-8 -*-
# @Time    : 2021/11/2 15:21
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : classification_model.py
# @Software: PyCharm
import torch
from torch import nn
import numpy as np
from torch.nn import functional
from torch.utils.data import DataLoader
from twitter_datasets import TwitterDatasets


class TextCNN(nn.Module):
    def __init__(self, num_embeddings):
        super(TextCNN, self).__init__()
        # Embedding的输入是 : [batch_size, seq_len] = 128 * 30
        # 输出 :  [batch_size, seq_len, embedding_dim]
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=64)
        # in_channels 这个是单个句子的长度 seq_len
        # in_channels : 30 , embedded = [128, 30, 64]
        # out_channels : [128, 32, 62]
        self.conv1d = nn.Conv1d(in_channels=30, out_channels=32, kernel_size=3, stride=2)
        # kernel_size 对应的是最后一维, 获取每一行的特征中的最大值
        self.pool = nn.MaxPool1d(kernel_size=31, stride=1)
        self.decoder = nn.Linear(32, 13) # [128, 2]
        self.softmax = nn.Softmax(dim=1) # out = [128,32] , dim = 1 表示 对 32 那一维的数据进行softmax

    def forward(self, inputs):
        # inputs : [128, 30] , embedded : [128, 30, 64]

        embedded = self.embedding(inputs)  # [batch_size, seq_len, embedding_dim]
        out = self.conv1d(embedded)
        out = self.pool(out)
        # out = torch.squeeze(out, -1)
        out = torch.reshape(out, [out.size()[0], -1])
        out = self.decoder(out)
        out = self.softmax(out)
        return out


if __name__ == '__main__':
    datasets = TwitterDatasets()
    train_loader = DataLoader(datasets, batch_size=128, shuffle=True)
    model = TextCNN(len(datasets.vocab_dict))

    for i, batch in enumerate(train_loader):
        data = batch[1]
        label = batch[0]

        out = model.forward(data)
        print(out)
        pred = torch.argmax(out, dim=1) # 获取指定维度上最大值的索引 : [128, 2] => dim = 1 的话就是 获取 [0.3, 0.6] 这一行数据数据中最大值的索引 => 1
        print(pred)
