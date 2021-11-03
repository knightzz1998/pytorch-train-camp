# -*- coding: utf-8 -*-
# @Time    : 2021/11/3 14:51
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : run_train.py
# @Software: PyCharm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from twitter_datasets import TwitterDatasets
from classification_model import TextCNN


def run():
    datasets = TwitterDatasets()
    train_loader = DataLoader(datasets, batch_size=128, shuffle=True)
    model = TextCNN(len(datasets.vocab_dict))
    # 将模型数据放到GPU上
    model.to(torch.device("cuda"))

    # 创建损失函数
    loss_func = nn.CrossEntropyLoss()

    # 定义优化算法
    optimizer = optim.Adam(params=model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.9
    )

    epochs = 30

    for epoch in range(epochs):

        for i, batch in enumerate(train_loader):
            label, data = batch

            # 将数据放到GPU上
            data = torch.tensor(data).to("cuda")
            label = torch.tensor(label).to("cuda")

            # 初始化梯度
            optimizer.zero_grad()

            # 获取预测值
            out = model.forward(data)

            # 计算loss
            loss_val = loss_func(out, label)

            # 计算accuracy
            pred = torch.argmax(out, dim=1)
            result = torch.eq(pred, label)
            accuracy = result.sum() * 1.0 / pred.size()[0]

            print("epoch is {}, val is {}, accuracy is {}".format(epoch, loss_val, accuracy))

            # 反向传播
            loss_val.backward()

            # 更新参数
            optimizer.step()
        scheduler.step()
        # 保存模型
        if epoch % 5 == 0:
            torch.save(model.state_dict(), "models/{}.pth".format(epoch))


if __name__ == '__main__':
    run()
