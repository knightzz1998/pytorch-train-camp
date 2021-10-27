### DataSet详解

#### Dataset数据读取流程

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


### DataLoader

#### 使用方法

```python
dataloader = DataLoader(dataset, batch_size=64, num_workers=3)
```

#### 参数详解

- batch_size  : 一个batch的数据大小
- num_workers : 多线程的线程数
- shuffle : 是否对数据shuffle
