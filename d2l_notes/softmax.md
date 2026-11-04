首先相比于线性回归而言，softmax不同点在于把原本是数字的自变量w,b变成了矩阵，对矩阵求导需要对每一项求偏导，而这同时会带来矩阵的相关运算

同时softmax设置了一系列对数据的处理操作，以人为构造“概率”

我们先来探究究竟做了哪些数学操作：

1.首先我们用于训练的数据是(256.784)的矩阵，代表256张图片每一张784个像素，每个像素对应一个亮度

2.我们现在需要w,b的初始值，即(784,10)的矩阵和(10,)的数组

3.首先(256,784)*(784,10)+(10,),这里的逻辑是将每一张图片的所有像素乘以对应的w求和再加上b，其中b会触发广播机制，最终得到(256,10),但目前这和概率并没有任何关系，因此我们选择通过指数变换得到和为1的概率形式，这种方式的好处是不仅可以转换为概率的形式，还保证了函数的可导性

4.接下来我们要计算Loss，

5.写到这里，我们回忆一下我们现在要做什么：我们要求Loss分别对w,b的导数

想要搞明白接下来的所有流程，我们不妨缩减一下数据量，不妨只有一张图片，两个像素，两个类别，我们看看整个过程发生了什么
<img width="614" height="145" alt="image" src="https://github.com/user-attachments/assets/f72fc7e9-0df5-434d-bd00-3950aaebf323" />
<img width="409" height="108" alt="image" src="https://github.com/user-attachments/assets/915a4066-16ba-4fc0-9eeb-61ca5d40d09c" />
<img width="470" height="101" alt="image" src="https://github.com/user-attachments/assets/1160a71f-fc0e-4839-ae3d-4cca9c1e9cda" />

<img width="246" height="50" alt="image" src="https://github.com/user-attachments/assets/05d86055-cab1-4bbb-bbe2-3ed15c012b01" />

这里就解释了接下来数据的变化情况



