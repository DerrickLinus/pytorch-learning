import torchvision
import os
from torch import nn

# 设置模型保存路径
os.environ['TORCH_HOME'] = 'd:/DL_projects/01_learn_pytorch/model_weights'

# dataset = torchvision.datasets.ImageNet("./dataset_ImageNet", split='train', transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(weights=None)
vgg16_True = torchvision.models.vgg16(weights='IMAGENET1K_V1')
# print("ok")
print(vgg16_True)

train_data = torchvision.datasets.CIFAR10('dataset_CIFAR10', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
# vgg16_True.add_module('add_linear', nn.Linear(1000,10))
vgg16_True.classifier.add_module('add_linear', nn.Linear(1000,10))
print(vgg16_True)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096,10)
print(vgg16_false)
