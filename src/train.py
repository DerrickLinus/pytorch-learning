import torch
import torchvision
from torch.utils.data import DataLoader
from model_Dlh import *
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import time

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
dlh = Dlh()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(dlh.parameters(), lr = learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10
# 添加Tensorboard
writer = SummaryWriter("./logs_train")

start_time = time.time()
for i in range(epoch):
    print(f"------------第 {i+1} 轮训练开始------------")
    
    # 训练步骤开始 
    dlh.train() # 在此模型中非必要 如有dropout、batchnorm层需要设置
    for data in train_dataloader:
        imgs, targets = data
        outputs = dlh(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器调优
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:  
            end_time = time.time()
            print(f"time：{end_time - start_time}")
            print(f"训练次数：{total_train_step}, loss：{loss.item()}")
            writer.add_scalar("train_loss", loss, total_train_step) 
    
    # 测试步骤开始
    dlh.eval() # 在此模型中非必要 如有dropout、batchnorm层需要设置
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = dlh(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
            
    print(f"整体测试集上的loss：{total_test_loss}")
    print(f"整体测试集上的正确率：{total_accuracy/test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1
    
    torch.save(dlh, "dlh_{}.pth".format(i))
    # torch.save(dlh.state_dict(), "dlh_{}.pth".format(i))
    print("模型已保存")

writer.close()
