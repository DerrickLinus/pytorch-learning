import torch
from torch import nn

x  = torch.tensor([1,2,3], dtype=torch.float32)
y = torch.tensor([1,2,5], dtype=torch.float32)

# L1Loss
loss = nn.L1Loss(reduction = "mean")
loss2 = nn.L1Loss(reduction = "sum")
result = loss(x,y)   # ((1-1)+(2-2)+(3-5))/3 = 0.667
result2 = loss2(x,y) # (1-1)+(2-2)+(3-5) = 2.000
print(f"loss of mean：{result:.3f}")
print(f"loss of sum：{result2:.3f}")

# MSELoss
loss_mse = nn.MSELoss()
result_mse = loss_mse(x,y)
print(f"result_mse：{result_mse:.3f}") # ((1-1)^2+(2-2)^2+(3-5)^2)/3 = 1.333

# CrossEntropyLoss 
# 交叉熵损失函数的计算公式：-\sum{y_i*log(y^hat)} y_i为真实标签，y^hat为预测值 (1)
# 计算时，不是乘以y_i的值，是将y_i作为索引，找到对应位置的y^hat，带入公式中求解

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x,(1,3))
loss_cross = nn.CrossEntropyLoss() # 实际计算公式：-x_target + log(exp(x_0)+exp(x_1)+...+exp(x_n)) (2)
print(f"result_cross：{result_cross}")
# (1)、(2)两种公式计算的结果差别不大，但(2)公式和程序运行的结果相同

# # Example of target with class indices
# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# output = loss(input, target)
# output.backward()

# # Example of target with class probabilities
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randn(3, 5).softmax(dim=1)
# output = loss(input, target)
# output.backward()

