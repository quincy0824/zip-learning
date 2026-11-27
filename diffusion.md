<img width="1020" height="565" alt="image" src="https://github.com/user-attachments/assets/fdecbee2-f3de-4ff2-81a2-0ba3f06296c5" />
<img width="1023" height="556" alt="image" src="https://github.com/user-attachments/assets/3e6b4984-0287-435c-9dd9-4da6324aacfe" />
<img width="1408" height="910" alt="image" src="https://github.com/user-attachments/assets/c50c410f-1044-45f3-8d7f-4dc42d7263ab" />
<img width="1053" height="890" alt="image" src="https://github.com/user-attachments/assets/45669efa-d8d2-4c5c-a43a-f4e855ad3e2c" />
diffusion如果用于生成图片，本质原因是因为真实数据往往不太够用，同时模拟生成的数据有些时候反倒会使模型的训练效果更好，因此我们使用diffusion来模拟生成图片。

生成的方法为：对于真图多轮施加不同的程度的噪声，模型通过噪声级别和脏图使用MLP反向猜测噪声，通过反向传播修改权重参数
当我们有了权重后，随便拿一张完全是噪声的图片，逐级反向抹除噪声，最终得到答案。




