# 池化层的作用：降维（保留主要特征），减少特征图的尺寸，降低计算复杂度

import torch
from torch import nn
from torch.nn import MaxPool2d
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1], 
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32) # 可以不加dtype=torch.float32

# input = torch.reshape(input, (-1, 1, 5, 5))

dataset = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, 64)

class Dlh(nn.Module):
    def __init__(self):
        super(Dlh, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False) # ceil_mode=True or False
    
    def forward(self, input):
        output = self.maxpool1(input)
        return output
 
dlh = Dlh()
# print(dlh)
# output = dlh(input)
# print(output)

writer = SummaryWriter("board_maxpool")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = dlh(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
