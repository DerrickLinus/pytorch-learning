# 线性层（全连接层）：进行线性变换 y=xW^T+b
# 作用：
# 将输入特征空间映射到新的特征空间
# 通过训练可以学习到输入特征之间的线性关系
# 通常与其他层（如激活函数）配合使用，构建更复杂的神经网络

import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Linear

dataset = torchvision.datasets.CIFAR10("./dataset_CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, 64, drop_last=True)

class Dlh(nn.Module):
    def __init__(self):
        super(Dlh, self).__init__()
        self.linear1 = Linear(196608,10)
    def forward(self, input):
        output = self.linear1(input)
        return output

dlh = Dlh()

for data in dataloader:
    imgs, targets = data
    # print(imgs.shape) # torch.size([64,3,32,32])
    output = torch.flatten(imgs)
    print(output.shape) # torch.size([196608])
    output = dlh(output)
    print(output.shape)