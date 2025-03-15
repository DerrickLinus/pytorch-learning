import os

root_dir = 'dataset_custom/train'

ants_target_dir = 'ants_image'
ants_img_path = os.listdir(os.path.join(root_dir, ants_target_dir))
ants_label = ants_target_dir.split('_')[0]
ants_out_dir = 'ants_label'

bees_target_dir = 'bees_image'
bees_img_path = os.listdir(os.path.join(root_dir, bees_target_dir))
bees_label = bees_target_dir.split('_')[0]
bees_out_dir = 'bees_label'


for i in ants_img_path:
    filename = i.split('.jpg')[0]
    with open(os.path.join(root_dir, ants_out_dir, filename + '.txt'), 'w') as f:
        f.write(ants_label)
    
for i in bees_img_path:
    filename = i.split('.jpg')[0]
    with open(os.path.join(root_dir, bees_out_dir, filename + '.txt'), 'w') as f:
        f.write(bees_label)        
        
