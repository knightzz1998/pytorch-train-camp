# -*- coding: utf-8 -*-
# @Time    : 2021/10/28 15:38
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : cnn_model.py
# @Software: PyCharm

from torch import nn
from torch.nn import Sequential, Conv2d, Linear, Module
import torchkeras

"""
    TextCNN + Word2Vec

"""


class TextCNN(Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.features = Sequential(
        )

    def forward(self, x):
        pass
