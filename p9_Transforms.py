from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
# 用法 
# Tensor数据类型
# 通过transforms.ToTensor去看两个问题
# 1. transforms该如何使用(python)
img_path = 'dataset_custom/train/ants_image/0013035.jpg'
img = Image.open(img_path)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
# print(tensor_img)

writer = SummaryWriter('log')
writer.add_image('Tensor_img', tensor_img)
writer.close()


# 2. 为什么我们需要Tensor数据类型
# Tensor是专门为深度学习设计的数据结构，支持在GPU上进行并行运算，比普通的Python列表或Numpy数组计算速度更快
# Tensor支持自动计算梯度（auto-grad），这是深度学习中反向传播的基础
# Pytorch的所有操作和神经网络层都是基于Tensor设计的
# 展示Tensor的一些重要特性
# print(f"Tensor的形状：{tensor_img.shape}") # 显示维度信息
# print(f"Tensor的数据类型：{tensor_img.dtype}") # 显示数据类型
# print(f"Tensor是否支持梯度计算：{tensor_img.requires_grad}") # 显示是否支持梯度计算
# print(f"Tensor的设备：{tensor_img.device}") # 显示tensor所在设备（CPU或GPU）
# # 启用梯度计算的Tensor示例
# import torch
# x = torch.tensor([1.], requires_grad=True)
# y = x*2
# y.backward()
# print(f"x的梯度：{x.grad}") # 显示自动计算的梯度