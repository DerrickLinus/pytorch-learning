import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_dataset = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_dataloader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=64, num_workers=0, drop_last=True)
# shuffle：是否打乱顺序 batch_size：一次取多少张图片进行打包 num_workers：进程 drop_last：是否舍弃未能打包成一个完整batch_size的图片
print(len(test_dataset))

# 查看测试集中的单张图片
img, target = test_dataset[0]
print(img.shape)
print(target)

# # 使用DataLoader
# for data in test_dataloader:
#     imgs, targets = data
#     print(imgs.shape) # 注意，这个地方需要在设置中对Terminal Scrollback的行数进行设置，否则在终端中显示不完全
#     print(targets)

# 设置轮次epoch
writer = SummaryWriter("board_dataloader")
for epoch in range(2):
    step = 0
    for data in test_dataloader:
        imgs, targets = data
        writer.add_images(f"Epoch：{epoch}", imgs, step)
        step = step + 1