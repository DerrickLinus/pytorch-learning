from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

img = Image.open('dataset_custom/train/ants_image/0013035.jpg')
# print(img)
writer = SummaryWriter('logs')

# ToTensor的使用
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)
writer.add_image('ToTensor', img_tensor)

# Normalize的使用
print(img_tensor[0][0][0]) # 查看第一个通道的第一个像素点的值
trans_norm = transforms.Normalize([0.5, 0.5 ,0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0]) # 查看经过归一化后，第一个通道的第一个像素点的值
writer.add_image('Normalize', img_norm, 2)

# Resize的使用
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img) # PIL -> PIL
print(img_resize)
img_resize = trans_tensor(img_resize) # PIL -> Tensor
print(img_resize)
writer.add_image('Resize', img_resize, 0)

#  Compose的使用 resize 2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_tensor]) # PIL -> PIL -> Tensor
img_resize_2 = trans_compose(img)
writer.add_image('Resize', img_resize_2, 1)

# RandomCrop的使用
# trans_random = transforms.RandomCrop(512)
trans_random = transforms.RandomCrop((500,700))
trans_compose_2 = transforms.Compose([trans_random, trans_tensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    # writer.add_image('RandomCrop', img_crop, i)
    writer.add_image('RandomCropHW', img_crop, i)

writer.close()