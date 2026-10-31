避免求和等操作对矩阵的降维：

```Python
sum_A = A.sum(axis=1, keepdims=True) # axis=1代表对矩阵横向求和，keepdims=True目的在于保持原有维数不变，便于广播时复制
tensor([[ 6.],
        [22.],
        [38.],
        [54.],
        [70.]])
```

矩阵向量积：

```Python
torch.mv(A, x)
```

矩阵乘积

```Python
torch.mm(A, B)
```

