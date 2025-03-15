import torch
from torch.nn import functional as F

# 张量（Tensor）是深度学习中的基本数据结构
# 0维张量：标量（单个数据）
# 1维张量：向量（一维数组）
# 2维张量：矩阵（二维数组）
# 3维及以上：高维张量

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]]) 

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = torch.reshape(input, (1, 1, 5, 5)) # [1批次, 1通道, 5高度, 5宽度] 
kernel = torch.reshape(kernel, (1, 1, 3, 3)) # [1输出通道, 1输入通道, 3高度, 3宽度]

print(input.shape) # shape中的数字个数=张量的维度
print(kernel.shape)

output = F.conv2d(input, kernel, stride=1)
print(output)
print(output.shape)

output2 = F.conv2d(input, kernel, stride=2)
print(output2)

output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)