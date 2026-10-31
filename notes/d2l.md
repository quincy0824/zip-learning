我们现在需要一个将数据清洗后传给深度学习网络的方法

1：创建人工数据集

```Python

import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)  # 表示找到上一级目录创建data文件夹

data_file = os.path.join('..', 'data', 'house_tiny.csv') # 搞出文件地址

with open(data_file, 'w') as f: # 'w'表示删去文件中所有内容重新开始，with用于自动关闭程序

    f.write('NumRooms,Alley,Price\n')  # 列名 一定要按格式写，否则pandas识别不了
    
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    
    f.write('2,NA,106000\n')
    
    f.write('4,NA,178100\n')
    
    f.write('NA,NA,140000\n')
```
    
2：使用pandas工具转变为表格形式

```Python

import pandas as pd

data = pd.read_csv(data_file) # data出厂时会被赋予一些工具
```

3：处理只含缺失值列

```Python

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2] # iloc用于列数索引

inputs = inputs.fillna(inputs.mean()) # mean()用于取平均值，fillna用于填充，但只能填充缺失值，pave之类的字符串填充不了，且以列为填充单位
```

4.处理不只含缺失值列

```Python

inputs = pd.get_dummies(inputs, dummy_na=True)

   NumRooms  Alley_Pave  Alley_nan # 这里会将每一个字符串单独列出来
   
0       3.0           1          0

1       2.0           0          1

2       4.0           0          1

3       3.0           0          1
```

5.转换为张量格式

```Python

import torch

X = torch.tensor(inputs.to_numpy(dtype=float)) # torch只接受浮点数，先要转成to_numpy，再转成tensor

y = torch.tensor(outputs.to_numpy(dtype=float))
```



