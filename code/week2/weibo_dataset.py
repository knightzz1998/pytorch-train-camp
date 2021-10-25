# -*- coding: utf-8 -*-
# @Time    : 2021/10/25 16:19
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : dataloader_demo.py
# @Software: PyCharm

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co

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
"""


class WeiBoDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index):
        pass
