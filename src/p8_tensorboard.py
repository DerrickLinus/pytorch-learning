from torch.utils.tensorboard import SummaryWriter
import cv2

writer = SummaryWriter('log')
img_path = 'dataset/train/ants_image/0013035.jpg'
# img = Image.open(img_path)
# img_array = np.array(img)
img = cv2.imread(img_path)
print(type(img))
print(img.shape)
writer.add_image('test', img, 1, dataformats='HWC')

for i in range(100):
    writer.add_scalar('y=2x', 2*i, i)

writer.close()    

# 在终端中打开该事件文件‘log'
# tensorboard --logdir=log
# 指定端口：tersorboard --logdir=log --port=6007
