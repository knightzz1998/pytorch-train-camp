# -*- coding: utf-8 -*-
# @Time    : 2021/10/25 21:14
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : weibo_data_split.py
# @Software: PyCharm

import pandas as pd
import os
from sklearn.model_selection import train_test_split


def read_data(data_dir):
    """
        读取数据
    :param data_dir:
    :return:
    """
    data = pd.read_csv(data_dir)
    X = data['review']
    y = data['label']
    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    train_data = pd.DataFrame({'label': y_train, 'review': X_train})
    test_data = pd.DataFrame({'label': y_test, 'review': X_test})

    return train_data, test_data


def save_data(train_data, test_data, train_path, test_path):
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)


if __name__ == '__main__':
    base_path = "../../datasets"
    data_dir = os.path.join(base_path, "weibo_senti_100k.csv")
    train_path = os.path.join(base_path, "weibo_train_data.csv")
    test_path = os.path.join(base_path, "weibo_test_data.csv")

    X, y = read_data(data_dir)
    train_data, test_data = split_data(X, y)
    save_data(train_data, test_data, train_path, test_path)
