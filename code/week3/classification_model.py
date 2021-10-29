# -*- coding: utf-8 -*-
# @Time    : 2021/10/28 10:29
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : classification_model.py
# @Software: PyCharm

"""
    1. Class WeiboModule(nn.Module) 实例 => weibo_module = WeiboModule()
    2. weibo_module(x) 这种调用形式实际上是调用了 父类的 __call__ 函数 参考 :http://c.biancheng.net/view/2380.html
    3. __init__ 构建子模块
    4. forward 拼接子模块

    - nn.Module
        - parameters:存储管理nn.Parameter类modules :存储管理nn.Module类
        - buffers :存储管理缓冲属性，如BN层中的running_mean
        - ***_hooks:存储管理钩子函数

    - 模型容器
"""

import torchvision

torchvision.models.AlexNet