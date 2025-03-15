from torch.utils.data import Dataset
from PIL import Image
import os
# img_path = 'dataset/train/ants/0013035.jpg'
# img = Image.open(img_path)
# print(type(img))
# print(img.size)
# img.show()
# import pdb 用于在终端查看变量
# import cv2

class mydata(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)
    
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)    
        label = self.label_dir.split('_')[0]
        return img, label
    
    def __len__(self):
        return len(self.img_path)
    

root_dir = 'dataset_custom/train'
ants_label_dir = 'ants_image'
bees_label_dir = 'bees_image'
ants_dataset = mydata(root_dir, ants_label_dir)
bees_dataset = mydata(root_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset