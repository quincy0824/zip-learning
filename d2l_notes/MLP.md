单纯的线性回归对于预测任务的准确率还是太低了，我们需要更复杂的模型以提高准确率，因此我们引进多层感知机

其本质是两层线性回归中间夹着一次较为复杂的函数映射，以提高模型的复杂程度并防止其退化为线性回归，而大多数现代神经网络使用的是f(x)=max{x,0}

```Python
import torch
from torch import nn

batch_size = 256
X_train = torch.randn(batch_size, 1, 28, 28) # 伪造256张图片
y_train = torch.randint(0, 10, (batch_size,)) # 伪造256个标签

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs))

params = [W1, b1, W2, b2]

def relu(X): # 这里就是中间层的函数
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)  
    return (H @ W2 + b2)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params, lr=0.1)

for epoch in range(5):
    optimizer.zero_grad()
    y_hat = net(X_train)
    l = loss_fn(y_hat, y_train)
    l.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {l.item():.4f}")
```
