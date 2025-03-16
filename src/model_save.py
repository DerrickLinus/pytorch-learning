import torch
import torchvision
from torch.serialization import safe_globals
from torchvision.models.vgg import VGG
# from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Linear, AdaptiveAvgPool2d, Dropout
from torch import nn

vgg_16 = torchvision.models.vgg16(weights=None)
# print(vgg_16)

# 保存方式1 保存模型结构+模型参数
torch.save(vgg_16, "vgg16_method1.pth")

# 保存方式2 保存模型参数（官方推荐）
torch.save(vgg_16.state_dict(), "vgg16_method2.pth")

class Dlh(nn.Module):
    def __init__(self):
        super(Dlh, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
    def forward(self, x):
        x = self.conv1(x)
        return x

dlh = Dlh()
torch.save(dlh, "dlh_method1.pth")