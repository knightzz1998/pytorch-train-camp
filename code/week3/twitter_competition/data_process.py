# -*- coding: utf-8 -*-
# @Time    : 2021/11/2 14:32
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : data_process.py
# @Software: PyCharm


import pandas as pd
from collections import Counter
from tqdm import tqdm


def read_data(data_path, data_type):
    data = pd.read_csv(data_path)

    if data_type == "train":
        labels = data['label'].values.tolist()
    else:
        labels = []

    contents = data['content'].values.tolist()
    return labels, contents


def read_stop_word(data_path):
    stop_words = open(data_path, "r", encoding="utf-8").readlines()
    # 去除空格
    stop_words = [word.strip() for word in stop_words]
    return stop_words


def data_process(contents, stop_words):
    # 保存去除停用词后的词表
    vocab_list = []

    # 存储句子中所有单词
    word_bags = []

    # 对数据分词
    for content in tqdm(contents):
        # 分词
        word_list = content.strip().split(" ")
        # 去除停用词
        seq_list = []
        for word in word_list:
            if word in stop_words:
                continue
            seq_list.append(word)
            word_bags.append(word)
        vocab_list.append(seq_list)

    # 统计词频获取不重复的单词表
    word_count_dict = Counter(word_bags)
    vocabs = [vocab for vocab in word_count_dict.keys()]

    return vocab_list, vocabs


def run_data_process(data_type):
    data_path = "datasets/train.csv"
    stop_words_path = "datasets/baidu_stopwords.txt"

    labels, contents = read_data(data_path, data_type)
    stop_words = read_stop_word(stop_words_path)

    vocab_list, vocabs = data_process(contents, stop_words)
    # print(vocab_list[:10])
    # print(vocabs[:10])
    return vocab_list, vocabs, labels


def run_test_data():
    data_path = "datasets/test.csv"
    stop_words_path = "datasets/baidu_stopwords.txt"

    data = pd.read_csv(data_path)
    contents = data['content'].values.tolist()
    stop_words = read_stop_word(stop_words_path)

    vocab_list, vocabs = data_process(contents, stop_words)
    return vocab_list, vocabs


if __name__ == '__main__':
    run_data_process()
