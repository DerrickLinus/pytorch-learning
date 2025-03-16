import torch
import torchvision
import model_save # 加载自定义模型时需要导入model_save.py

# 加载方式1（完全信任模型文件来源的情况下使用）
model1 = torch.load("vgg16_method1.pth", weights_only=False)
# print(model1)

model2 = torch.load("vgg16_method2.pth", weights_only=False) 
# print(model2) # 模型参数以字典形式

# 加载方式2
vgg_16 = torchvision.models.vgg16(weights=None)
vgg_16.load_state_dict(torch.load('vgg16_method2.pth')) # 将参数放进模型中
# print(vgg_16)

# 加载方式1 加载自定义模型 需要导入model_save.py
model3 = torch.load("dlh_method1.pth", weights_only=False)
print(model3)