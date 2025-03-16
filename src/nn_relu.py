# 非线性变换：使神经网络能够拟合任意复杂的函数

import torch
from torch import nn
from torch.nn import ReLU
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Sigmoid

# input = torch.tensor([[1, -0.5],
#                      [-1, 2]])
# print(input.shape)

dataset = torchvision.datasets.CIFAR10("./dataset_CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, 64)

class Dlh(nn.Module):
    def __init__(self):
        super(Dlh,self).__init__()
        self.relu1 = ReLU() # 将负值置为0，正值保持不变，计算简单且能有效缓解梯度消失问题
        self.sigmoid = Sigmoid() # 将输入压缩到(0,1)之间，常用于二分类问题的输出层
        
    def forward(self, input):
        # output = self.relu1(input)
        output = self.sigmoid(input)
        return output
    
dlh = Dlh()
# output = dlh(input)
# print(output)

writer = SummaryWriter("board_sigmoid")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = dlh(imgs)
    writer.add_images("output", output, step)
    step = step + 1
    
writer.close()
