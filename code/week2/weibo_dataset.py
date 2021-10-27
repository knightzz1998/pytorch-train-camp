# -*- coding: utf-8 -*-
# @Time    : 2021/10/25 16:19
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : dataloader_demo.py
# @Software: PyCharm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import jieba

"""
    torch.utils.data.DataLoader
    - 功能 : 创建可以迭代的数据装载器
    - 参数 :
        1. dataset : Dataset类对象, 决定数据从哪读取以及如何读取
        2. batchsize: 决定数据批次大小
        3. num_works: 多进程读取数据的线程数
        4. shuffle: 每个 epoch 是否乱序
        5. 当样本数不能被batchsize整除时, 是否舍去最后一个batch的数据
    - 名词解释 :
        1. 样本总数 : 80, batchsize : 8 => 1 Epoch = 10 iteration

    torch.utils.data.Dataset
    - 功能 : Dataset抽象;类, 所有自定义的Dataset都需要继承他, 并重写相应的方法
    - __getitem__(self, index)
        1. 接收一个索引, 返回一个样本 : index => label, data
        2. 返回的样本的大小要一样
"""


class WeiBoDataset(Dataset):

    def __init__(self, data_path):
        # 读取数据
        self.label, self.data = self.read_data(data_path)

    def __len__(self):
        """
            这个必须要设置, getitem中的index就是根据这个来设置的
        :return:
        """
        return len(self.data)

    def __getitem__(self, index):
        """
            1. 先获取数据集的大小 __len__
            2. 根据len生成index, 然后shuffle
            3. 根据shuffle后的数据以及batch_size生成索引列表batch_index, 索引列表的大小为 batch_size
            4. 获取每个batch的数据时, 根据batch_index传入到 __getitem__ 获取对应的数据
            5. 注意 : batch的数据类型取决于__getitem__返回的类型, 一般都会转换为tensor
            6. 有的数据类型是无法转换为tensor的, 比如 元素类型为str的list
            7. default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists found
            8. 上面报错原因就是 因为数据无法转换为 tensor , 而类型又不属于 tensors, numpy arrays, numbers, dicts or lists 这几种
            9. 如果返回的数据是集合类型, 可以直接使用 np.array() 转换为ndarray类型, 这样会被自动转换为tensor, 当然要求这个集合类型的元素类型是tensor有的
            10. 如果是tensor没有的,比如 str 类型的, 反而会报错, 比如 7. 报错
        :param index:
        :return:
        """
        label = 1
        # features = np.array([str(i) for i in range(10)])
        features = np.array([i for i in range(10)])
        return label, features

    @staticmethod
    def read_data(data_path):
        data = pd.read_csv(data_path)
        return data['label'], data['review']


if __name__ == '__main__':
    weibo_dataset = WeiBoDataset("../../datasets/weibo_test_data.csv")
    dataloader = DataLoader(weibo_dataset, batch_size=1024, shuffle=True)
    for i, batch in enumerate(dataloader):
        print(type(batch[0]), type(batch[1]))
