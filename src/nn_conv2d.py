# 卷积层的作用：特征提取

import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(
    root="./dataset_CIFAR10", 
    train=False, 
    transform=torchvision.transforms.ToTensor(),  # 添加括号()使transform生效
    download=True
)

dataloader = DataLoader(dataset, batch_size=64)

class Dlh(nn.Module):
    def __init__(self):
        super(Dlh, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        
    def forward(self, x):
        x = self.conv1(x)
        return x

dlh = Dlh()
print(dlh)

writer = SummaryWriter("board_conv2d")
step = 0
for data in dataloader:
    imgs, targets = data
    output = dlh(imgs)
    # print(imgs.shape)
    # print(output.shape)
    writer.add_images("input", imgs, step) # torch.Size([64, 3, 32, 32])
    # writer.add_images("output", output, step) # torch.Size([64, 6, 32, 32]) tensorboard无法显示六通道
    output = torch.reshape(output, (-1,3,30,30))
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
    

