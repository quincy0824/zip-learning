预先设立w，b

根据超参数和随机数据算出标签

把随机数据和标签喂给程序以反向推算超参数，从而验证梯度下降的有效性

生成数据：
```Python
import torch
import random
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w))) # 生成随机数据，这里生成的是1000行2列的矩阵
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape) # 让数据有点波动
    return X, y.reshape((-1, 1))
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000) # 自此，我们所需要的数据构建完毕
def data_iter(batch_size, features, labels): # 构建小批量数据生成器
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) # 打乱顺序
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices] # yield只返回值而不结束函数
def linreg(X, w, b):
    return torch.matmul(X, w) + b
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2 # 计算误差值
def sgd(params, lr, batch_size):
    with torch.no_grad(): # 更新参数时不需要记录计算图
        for param in params:
            param -= lr * param.grad / batch_size # 在计算导数时把十组数据求和，因此需要求导数的平均值
            param.grad.zero_()
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.03           # 学习率
num_epochs = 3      # 把整个数据集跑 3 遍
batch_size = 10     # 每次抓 10 条数据
net = linreg
loss = squared_loss
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  
        l.sum().backward()         
        sgd([w, b], lr, batch_size) 
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
print("-" * 30)
print(f'真实 w: {true_w}, 真实 b: {true_b:.4f}')
print(f'学到的 w: {w.reshape(true_w.shape).detach().numpy()}, 学到的 b: {b.item():.4f}')

# 算一下误差
print(f'w 的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b 的估计误差: {true_b - b}')
```
