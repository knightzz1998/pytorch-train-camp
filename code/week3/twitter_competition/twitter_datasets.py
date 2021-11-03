# -*- coding: utf-8 -*-
# @Time    : 2021/10/28 15:58
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : datasets.py
# @Software: PyCharm
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from data_process import run_data_process


class TwitterDatasets(Dataset):

    def __init__(self, data_type="train") -> None:
        super(TwitterDatasets, self).__init__()
        self.data_type = data_type
        self.vocab_list, self.vocabs, self.labels = run_data_process(data_type)
        # 读取单词索引表
        self.vocab_dict = self.get_word2idx()
        self.max_seq_len = self.get_max_seq_len()

    def __len__(self):
        return len(self.vocab_list)

    def __getitem__(self, index):

        assert len(self.labels) == len(self.vocab_list)
        word_list = self.vocab_list[index]

        data = []
        # 将单词替换为索引
        for word in word_list:
            if word in self.vocab_dict:
                data.append(self.vocab_dict[word])
            else:
                data.append(self.vocab_dict['<PNK>'])
        # 对长度不够的数据进行padding
        if len(data) < self.max_seq_len:
            data += [self.vocab_dict['<PAD>'] for _ in range(self.max_seq_len - len(data))]

        data = np.array(data)

        if self.data_type == "train":
            label = self.labels[index]

        return label, data

    def get_word2idx(self):
        # 获取词表
        vocab_dict = {vocab: index for index, vocab in enumerate(self.vocabs)}
        # 添加PAD和UNK
        vocab_dict.update({'<PAD>': len(vocab_dict), "<UNK>": len(vocab_dict) + 1})
        return vocab_dict

    def get_max_seq_len(self):
        """
            获取最长的句子长度
        :return:
        """
        max_seq_len = 0
        for vocab in self.vocab_list:
            if max_seq_len < len(vocab):
                max_seq_len = len(vocab)
        return max_seq_len


if __name__ == '__main__':
    datasets = TwitterDatasets()
    train_loader = DataLoader(datasets, batch_size=128, shuffle=True)

    for i, batch in enumerate(train_loader):
        print(batch[1].size())
