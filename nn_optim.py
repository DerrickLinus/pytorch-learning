# 优化器
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
# 将模型移动到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dlh = dlh.to(device)
print(f"Using device: {device}")

loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(dlh.parameters(), lr=0.01) # 优化器 SGD随机梯度下降法

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        # 将数据移动到 GPU
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        output = dlh(imgs)
        result_loss = loss(output, targets)
        optim.zero_grad()
        result_loss.backward() # 反向传播
        optim.step()
        # print(result_loss)
        running_loss = running_loss + result_loss
    print(running_loss)
    
