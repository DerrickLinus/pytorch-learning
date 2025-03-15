# GPU加在模型、损失函数、数据（输入、标签）
import torch
import torchvision
from torch.utils.data import DataLoader
# from model_Dlh import * 
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.tensorboard import SummaryWriter
import time

# device = torch.device("cpu") # 使用CPU训练
# device = torch.device("cuda")  # 使用GPU训练
# 更专业的写法
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 准备训练数据集
train_data = torchvision.datasets.CIFAR10('./dataset_CIFAR10', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
# 准备测试数据集
test_data = torchvision.datasets.CIFAR10('./dataset_CIFAR10', train=False, transform=torchvision.transforms.ToTensor(),
                                          download=True)

# 查看数据集
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度为：{train_data_size}")
print(f"测试数据集的长度为：{test_data_size}")

# 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
class Dlh(nn.Module):
    def __init__(self):
        super(Dlh, self).__init__()
        self.model = nn.Sequential(
            Conv2d(3, 64, 3, 1, 1),  # 增加初始通道数
            nn.BatchNorm2d(64),      # 添加批归一化
            nn.ReLU(),               # 添加激活函数
            MaxPool2d(2),
            Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            MaxPool2d(2),
            Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            MaxPool2d(2),
            Flatten(),
            Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(0.5),        # 添加Dropout防止过拟合
            Linear(512, 10)
        )
        
    def forward(self,x):
        x = self.model(x)
        return x
    
dlh = Dlh()
# 使用GPU
dlh.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 使用GPU
loss_fn.to(device)

# 优化器
learning_rate = 1e-3 # 降低学习率
optimizer = torch.optim.Adam(dlh.parameters(), lr = learning_rate) # 使用Adam优化器

# 添加学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 20
# 添加Tensorboard 
writer = SummaryWriter("./logs_train")

start_time = time.time()
for i in range(epoch):
    print(f"------------第 {i+1} 轮训练开始------------")
    
    # 训练步骤开始 
    dlh.train() 
    for data in train_dataloader:
        imgs, targets = data
        # 使用GPU
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = dlh(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器调优
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:  
            end_time = time.time()
            print(f"time：{end_time-start_time}")
            print(f"训练次数：{total_train_step}, loss：{loss.item()}")
            writer.add_scalar("train_loss", loss, total_train_step)
    
    # 测试步骤开始
    dlh.eval() 
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            # 使用GPU
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = dlh(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
            
    print(f"整体测试集上的loss：{total_test_loss}")
    print(f"整体测试集上的正确率：{total_accuracy/test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    scheduler.step(total_test_loss) # 在每个epoch结束后更新学习率
    
    total_test_step = total_test_step + 1
    
    # 模型保存
    torch.save(dlh, "dlh_{}.pth".format(i))
    # torch.save(dlh.state_dict(), "dlh_{}.pth".format(i))
    print("模型已保存")

writer.close()
