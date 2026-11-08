首先相比于线性回归而言，softmax不同点在于把原本是数字的自变量w,b变成了矩阵，对矩阵求导需要对每一项求偏导，而这同时会带来矩阵的相关运算

同时softmax设置了一系列对数据的处理操作，以人为构造“概率”

我们先来探究究竟做了哪些数学操作：

1.首先我们用于训练的数据是(256,784)的矩阵，代表256张图片每一张784个像素，每个像素对应一个亮度

2.我们现在需要w,b的初始值，即(784,10)的矩阵和(10,)的数组

3.首先(256,784)*(784,10)+(10,),这里的逻辑是将每一张图片的所有像素乘以对应的w求和再加上b，其中b会触发广播机制，最终得到(256,10),但目前这和概率并没有任何关系，因此我们选择通过指数变换得到和为1的概率形式，这种方式的好处是不仅可以转换为概率的形式，还保证了函数的可导性

4.接下来我们要计算Loss，代码中为我们提供了一种方法，可以将(256,10)矩阵中对应真实答案的数值挑选出来，这样我们得到了256个数值，先对其分别求负对数，这里目的在于将偏离正确答案过多的数据放大，然后求和得到最终可反向传播的数值

5.写到这里，我们回忆一下我们现在要做什么：我们要求Loss分别对w,b的导数

想要搞明白接下来的所有流程，我们不妨缩减一下数据量，不妨只有一张图片，两个像素，两个类别，我们看看整个过程发生了什么

<img width="614" height="145" alt="image" src="https://github.com/user-attachments/assets/f72fc7e9-0df5-434d-bd00-3950aaebf323" />
<img width="409" height="108" alt="image" src="https://github.com/user-attachments/assets/915a4066-16ba-4fc0-9eeb-61ca5d40d09c" />
<img width="470" height="101" alt="image" src="https://github.com/user-attachments/assets/1160a71f-fc0e-4839-ae3d-4cca9c1e9cda" />


<img width="246" height="50" alt="image" src="https://github.com/user-attachments/assets/05d86055-cab1-4bbb-bbe2-3ed15c012b01" />


这里就解释了接下来数据的变化情况

6.转置后的原始数据为(784,256)，乘上dz后还是(256,10)的矩阵，变为(784,10)，正好和w的矩阵形状一致，而dz后的(256,10)矩阵，因为b在广播时复制256份，因此每列相加求平均得到(10,)，也正好与b的矩阵相同，至此，我们可以更新w,b并完成了一整轮的训练，下面是代码部分

```Python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size = 256
lr = 0.1
num_epochs = 10

transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_inputs = 784  # 28x28 的图片展开
num_outputs = 10  # 10 个类别
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X) 
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def net(X):
    X_flat = X.reshape((-1, W.shape[0]))
    return softmax(torch.matmul(X_flat, W) + b)

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

def evaluate_accuracy(data_iter, net):
    correct, total = 0, 0
    with torch.no_grad(): # 测试阶段不需要记录计算图
        for X, y in data_iter:
            y_hat = net(X)
            # argmax 找出概率最大的索引，如果等于真实标签 y，说明预测正确
            correct += (y_hat.argmax(axis=1) == y).sum().item()
            total += y.numel()
    return correct / total

for epoch in range(num_epochs):
    train_loss_sum = 0.0
    train_acc_sum = 0.0
    
    for X, y in train_iter:
        y_hat = net(X)
        l = cross_entropy(y_hat, y) 
        l.sum().backward() 
        
        with torch.no_grad():
            W -= lr * W.grad / batch_size
            b -= lr * b.grad / batch_size
            W.grad.zero_()
            b.grad.zero_()
            
        train_loss_sum += l.sum().item()
        train_acc_sum += (y_hat.argmax(axis=1) == y).sum().item()

    # 一个 Epoch 跑完，算平均指标并打印
    epoch_train_loss = train_loss_sum / len(train_dataset)
    epoch_train_acc = train_acc_sum / len(train_dataset)
    epoch_test_acc = evaluate_accuracy(test_iter, net)
    
    print(f"Epoch {epoch+1:2d} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Test Acc: {epoch_test_acc:.4f}")

print("训练结束！")
```





