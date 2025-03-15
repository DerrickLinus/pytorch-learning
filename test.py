from PIL import Image
import torchvision
import torch
from model_Dlh import *

# 读取待分类图片
image_path = "./Image/image.png"
image = Image.open(image_path)
# print(image)

# transforms变换
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)                                           
# print(image.shape)

# 加载自定义模型
model = torch.load("dlh_19.pth", weights_only=False)
# model = torch.load("dlh_19.pth", weights_only=False, map_location=torch.device("cpu"))
# print(model)
image = torch.reshape(image, (1, 3, 32, 32))
if torch.cuda.is_available():
    image = image.cuda()

# 测试模式
model.eval()
with torch.no_grad():
    output = model(image)
    
print(output)
print(output.argmax(1))

