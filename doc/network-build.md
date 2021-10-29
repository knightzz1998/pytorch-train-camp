### NN库结构



#### nn.Module

1. Class WeiboModule(nn.Module) 实例 => weibo_module = WeiboModule()
2. weibo_module(x) 这种调用形式实际上是调用了 父类的 __call__ 函数 : [参考文章](http://c.biancheng.net/view/2380.html)
3. __init__ 构建子模块)
4. forward 拼接子模块

- nn.Module
   - parameters:存储管理nn.Parameter类modules :存储管理nn.Module类
   - buffers :存储管理缓冲属性，如BN层中的running_mean
   - ***_hooks:存储管理钩子函数

#### Sequential

- nn.Sequential是nn.module的容器，用于按顺序包装一组网络层
- 顺序性:各网络层之间严格按照顺序构建
- 自带forward():自带的forward里，通过for循环依次执行前向传播运算

#### ModuleList

- nn.ModuleList是nn.module的容器，用于包装一组网络层,以迭代方式调用网络层
- append():在ModuleList后面添加网络层extend(): 拼接两个ModuleList
- insert():指定在ModuleList中位置插入网络层
- ModuleList 可以迭代的构建多层模型

```python
self.linears = nn.ModuleList( [nn.Linear(10, 10)for i in range(20)])
```

#### ModuleDict

- nn.ModuleDict是 nn.module的容器，用于包装一组网络层，以索引方式调用网络层主要方法:
1. clear(): 清空ModuleDict
2. items(): 返回可迭代的键值对(key-value pairs)
3. keys():返回字典的键(key)
4. values(): 返回字典的值(value)
5. pop(): 返回一对键值，并从字典中删除


```python
self.choices = nn. ModuleDict({
'conv': nn.conv2d(10, 10, 3),
'pool': nn.MaxPool2d(3)
})
```

- 这种结构的好处是可以传入key来选择使用的模型



### NN 库模型参数详解



#### 卷积操作输出形状计算公式

$$
output_shape = \dfrac{imageShape - filterShape + 2 * padding}{stride} + 1
$$



#### Conv2d

> 参考文章 : 
>
> - [torch.nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)
>
> - [PyTorch学习笔记(9)——nn.Conv2d和其中的padding策略](https://blog.csdn.net/g11d111/article/details/82665265)
> - [nn.Conv2d中padding详解【pytorch学习】](https://blog.csdn.net/a19990412/article/details/83904476)



- stride 卷积操作窗口移动的步长
- padding 的作用是补0, padding策略是把0补在左上。padding是在卷积之前补0
- dilation(扩张) :控制kernel点**卷积核点**的间距
- kernel_size: 卷积核的大小





### 卷积层运算

#### 卷积运算

- 卷积运算:卷积核在输入信号（图像)上滑动，相应位置上进行乘加

- 卷积核:又称为滤波器,过滤器，可认为是某种模式，某种特征。



![image-20211028120321341](https://haloos.oss-cn-beijing.aliyuncs.com/typero/image-20211028120321341.png)

- 卷积操作步骤
  - 先用卷积核 (一个窗口) 得到一个区域的值, 
  - 然后将这个区域的值和对应的参数值相乘然后求和得到一个值作为输出
  - 卷积操作时 n => 1 的操作





### AlexNet 模型构建



#### 模型结构图



![image-20211028112433071](https://haloos.oss-cn-beijing.aliyuncs.com/typero/image-20211028112433071.png)



### AlexNet 模型代码

- torchvision.models.AlexNet

```python
class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
```



#### AlexNet 构建步骤

- self.features = nn.Sequential
- Conv2d 的参数详解
  - stride 卷积操作窗口移动的步长

```python

# 添加一个 2维的卷积层

nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),


```









- - 