绘制函数图像模版：

```Python
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = np.sin(x)
h = 1e-5 
dy_dx = (np.sin(x + h) - np.sin(x - h)) / (2 * h)
plt.figure(figsize=(8, 5))
plt.plot(x, y, label="f(x) = sin(x)", color="blue")
plt.plot(x, dy_dx, label="derivative (calculated by code)", color="red", linestyle="--")
plt.grid(True) # 打开背景里的网格线
plt.legend() # 显示写过的label标签
plt.show()
```




  
