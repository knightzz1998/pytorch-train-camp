# -*- coding: utf-8 -*-
# @Time    : 2021/10/25 21:39
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : dataset_run.py
# @Software: PyCharm

from torch.utils.data import Dataset, DataLoader
from weibo_dataset import WeiBoDataset

if __name__ == '__main__':
    # 使用Dataloader读取数据
    data_path = "../../datasets/weibo_test_data.csv"
    dataset = WeiBoDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=3)
    for i, batch in enumerate(dataloader):
        print(i, batch)