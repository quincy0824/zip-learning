过拟合是指机器在使用训练数据训练时过于准确而导致在面对测试数据时准确度下降

想要解决这一问题，我们可以将Loss值变大从而调整模型，因此我们需要寻找一个只在过拟和中明显变大的变量。我们发现通常情况下w会在过拟合时为了meet噪音而被放大，所以一个解决办法是给Loss加上w的大小(当然为了简洁在此略做调整)

下面是唯一和正常MLP不同的地方
```Python
 trainer = torch.optim.SGD([
        {"params": net[0].weight, 'weight_decay': wd},
        {"params": net[0].bias}
    ], lr=lr)
```
