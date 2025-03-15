import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

dataset = torchvision.datasets.CIFAR10("./dataset_CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=1)

class Dlh(nn.Module):

    def __init__(self):
        super(Dlh, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5 ,padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5 ,padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
        
    def forward(self, input):
        x = self.model1(input)
        return x
    
dlh = Dlh()

# 损失函数的第一个作用：计算实际输出和目标之间的差距
loss = nn.CrossEntropyLoss()
for data in dataloader:
    imgs, targets = data
    output = dlh(imgs)
    result_loss = loss(output, targets)
    # print(output)
    # print(targets)
    # print(result_loss)
    result_loss.backward()
    # print("ok")