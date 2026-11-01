本质上是一个顺序记住链式法则，再反向求导的过程

```Python
import torch
x = torch.arange(4.0)
x.requires_grad_(True) # 顺序记忆
y = 2 * torch.dot(x, x) # 因为链式法则，所以按什么顺序返回追溯都一样
y.backward() 3 只能对数字溯源，溯源时最先对产生最终结果的式子求导，注意，所求函数是否含有while，if，else对回溯无影响
print(x.grad) # 这里想要查看最终结果要使用grad，因为在记忆的时候是把结果储存在x.grad里面的
x.grad.zero_() # 每一次求完导后都要清空，一边下一次使用
y = x * x
u = y.detach() # 用于斩断y与x的联系
z = u * x
z.sum().backward() # 如果print(x.grad == u)会发现都是true，同时这里要把最终结果变成数字后再向上溯源
x.grad.zero_()
```
