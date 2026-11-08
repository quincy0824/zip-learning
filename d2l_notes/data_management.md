在工程上，我们通常将每一层的代码封装为块

```Python
import torch
from torch import nn

class MyBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        spuer().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, out_dim)
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# 实例化模型
net = MyBlock(784, 10)
```
而在这里，我们有几种操作参数的手段

1.访问特定层
```Python
print(net.fc1.weight) # 直接点出来，打印的是 128x784 的矩阵
print(net.fc2.bias)
```

2.遍历所有参数
```Python
for name, param in net.named_parameters():
    print(f"参数名: {name}, 形状: {param.shape}")
```

3.自行给参数赋值(这里有两种流行方法，都是为了保证方差稳定，可以从参数初始化层面解决梯度消失和爆炸问题)

i：如果对隐藏层使用RELU函数，我们使用Kaiming初始化

```Python
# 方式 A：Kaiming 均匀分布（默认）
nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
# 方式 B：Kaiming 正态分布
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

ii：如果使用Sigmoid函数，可以使用Xavier初始化

```Python
# 方式 A：Xavier 均匀分布（默认范围内均匀采样）
nn.init.xavier_uniform_(layer.weight)
# 方式 B：Xavier 正态分布（均值为 0，根据输入输出维度计算标准差）
nn.init.xavier_normal_(layer.weight)
```

4.参数共享(这里有两种方法)

```Python
class SharedNet(nn.Module):
    def __init__(self):
        super().__init__()
        shared_layer = nn.Linear(128, 128)
         
        self.layer_a = shared_layer
        self.layer_b = shared_layer

    def forward(self, x):
        x = self.layer_a(x)
        x = self.layer_b(x) 
        return x
net = SharedNet()
```
