# -*- coding: utf-8 -*-
# @Time    : 2021/10/28 16:09
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : word2vce_data_process.py
# @Software: PyCharm

from torch.utils.data import DataLoader, Dataset
from gensim.models import Word2Vec
import pandas as pd

"""
    1. 读取训练集和测试集的数据来构建词向量

"""


class Word2vceTwitter():
    def __init__(self, train_path, test_path):
        self.data = self.read_data(train_path, test_path)

    def train_vec(self):
        """
            训练词向量
        :return:
        """
        pass

    @staticmethod
    def read_data(train_path, test_path):
        # 读取训练数据和测试数据的文本集
        train_data = pd.read_csv(train_path)['content']
        test_data = pd.read_csv(test_path)['content']
        # 合并文本
        data = pd.DataFrame([train_data, test_data])
        return data


if __name__ == '__main__':
    train_path = "datasets/train.csv"
    test_path = "datasets/test.csv"

    word2vec_twitter = Word2vceTwitter(train_path, test_path)
