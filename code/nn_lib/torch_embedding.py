# -*- coding: utf-8 -*-
# @Time    : 2021/11/1 16:28
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : torch_embedding.py
# @Software: PyCharm
import torch
from torch import nn
from torch.nn import Sequential, Conv1d, Linear, Module, Embedding, Dropout, functional
import torchkeras
from collections import Counter

"""
    1. 参数 : 
        - num_embeddings : 单词表(对文本去重后得到的词表) 的大小
        - embedding_dim : 每一个单词对应的向量大小
    
    2. 参考文章
        - https://www.jianshu.com/p/63e7acc5e890
"""


class EmbeddingModule(Module):

    def __init__(self, word_dict_len, word_dim, padding_idx=0):
        super(EmbeddingModule, self).__init__()
        self.embedding = Embedding(num_embeddings=word_dict_len, embedding_dim=word_dim, padding_idx=padding_idx)

    def forward(self, inputs):
        # Embedding层的输入形状为NxM（N是batch size,M是序列的长度），则输出的形状是 N * M * embedding_dim
        out = self.embedding(inputs)
        return out


if __name__ == '__main__':
    sentences = [
        "i love you",
        "i hate you",
        "the dog is black",
        "i like the cat"
    ]

    # 对每一行的数据进行分词
    # [['i', 'love', 'you'], ['i', 'hate', 'you'], ['the', 'dog', 'is', 'black'], ['i', 'like', 'the', 'cat']]
    word_list = [sentence.strip().split(" ") for sentence in sentences]
    print(word_list)

    # 统计词频
    word_list_count = Counter([word for words in word_list for word in words])
    print(word_list_count)

    # 获取单词词表
    vocabs = [word for word in word_list_count.keys()]

    # 获取单词索引
    # {'is': 0, 'love': 1, 'the': 2, 'like': 3, 'dog': 4, 'you': 5, 'black': 6, 'hate': 7, 'cat': 8, 'i': 9}
    vocabs_idx = {word: idx for idx, word in enumerate(vocabs)}

    # 添加 pad 和 unk, 作用是分别为了填充句子和替换词表中不存在的词
    vocabs_idx.update({'<PAD>': len(vocabs_idx), '<UNK>': len(vocabs_idx) + 1})
    print(vocabs_idx)

    # 将原本的单词转换为索引
    sentences_idx = []
    max_seq_len = 4
    for sentence in word_list:
        seq_res = []
        for word in sentence:
            if word in vocabs_idx.keys():
                seq_res.append(vocabs_idx[word])
            else:
                seq_res.append(vocabs_idx['<UNK>'])
        # 对数据进行padding , 保证每个句子长度一致
        if len(seq_res) < max_seq_len:
            seq_res += [vocabs_idx['<PAD>'] for _ in range(max_seq_len - len(seq_res))]
        sentences_idx.append(seq_res)

    # [[7, 3, 1], [7, 0, 1], [6, 8, 9, 5], [7, 2, 6, 4]]
    print(sentences_idx)

    # 注意 : tensor 转换需要每个数组的大小相同
    sentences_idx = torch.tensor(sentences_idx)

    model = EmbeddingModule(len(vocabs_idx), 5)
    out = model.forward(sentences_idx)

    print(sentences_idx.size())
    print(out.size())
