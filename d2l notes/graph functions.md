绘制函数图像模版：

1.新建文件
  
```Python
plot_math.py
```

2.复制生成图像程序代码

```Python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
f_expr_str = "x**3 - 1/x"   # 你要画的函数表达式（注意 Python 乘法用 *，乘方用 **）
x_start, x_end = -3, 3      # X 轴的起止范围
tangent_x0 = 1              # 你想画哪个点的切线（填 None 则不画切线）
x_sym = sp.Symbol('x')
f_sym = sp.sympify(f_expr_str)
df_sym = sp.diff(f_sym, x_sym)
f_num = sp.lambdify(x_sym, f_sym, 'numpy')
df_num = sp.lambdify(x_sym, df_sym, 'numpy')
x = np.linspace(x_start, x_end, 1000)
with np.errstate(divide='ignore', invalid='ignore'):
    y = f_num(x)
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', label=f"f(x) = {f_expr_str}")
if tangent_x0 is not None:
    k = df_num(tangent_x0)       
    fx0 = f_num(tangent_x0)      
    tangent_y = k * (x - tangent_x0) + fx0
    plt.plot(x, tangent_y, 'r--', label=f"Tangent at x={tangent_x0}")
    plt.scatter([tangent_x0], [fx0], color='black', zorder=5) 
    plt.title(f"Function and Tangent (Slope = {k:.2f})")
plt.axhline(0, color='black', linewidth=0.5) 
plt.axvline(0, color='black', linewidth=0.5) 
plt.grid(True, linestyle=':', alpha=0.7)    
plt.legend()                                
if np.any(np.isfinite(y)):
    valid_y = y[np.isfinite(y)]
    plt.ylim(np.percentile(valid_y, 5) - 1, np.percentile(valid_y, 95) + 1)
plt.show()
```

3.环境准备

```Python
miniconda3\Scripts\activate.bat zip-week1 # 前面要加东西
pip install numpy matplotlib sympy
```

4.输入指令

```Python
python plot_math.py
```
  
