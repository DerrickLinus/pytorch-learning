import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear

# 搭建神经网络
class Dlh(nn.Module):
    def __init__(self):
        super(Dlh, self).__init__()
        self.model = nn.Sequential(
            Conv2d(3, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, 2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
        
    def forward(self,x):
        x = self.model(x)
        return x

# 验证
if __name__ == "__main__":
    dlh = Dlh()
    input = torch.ones((64, 3, 32, 32))
    output = dlh(input)
    print(output.shape)
