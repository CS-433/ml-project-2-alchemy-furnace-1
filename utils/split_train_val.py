
# img_dirs = ['datasets/training/images_train', 'datasets/training/image_rotation', 'datasets/training/image_flip', 'datasets/training/image_rotation45']
# mask_dirs = ['datasets/training/groundtruth_binary_train', 'datasets/training/groundtruth_rotation', 'datasets/training/groundtruth_flip', 'datasets/training/groundtruth_rotation45']
# img_dirs = ['datasets/training/image_rotation45v3']
# mask_dirs = ['datasets/training/groundtruth_rotation45v3']
img_dirs = ['datasets/training/image_rotation', 'datasets/training/image_flip', 'datasets/training/image_rotation45']
mask_dirs = ['datasets/training/groundtruth_rotation', 'datasets/training/groundtruth_flip', 'datasets/training/groundtruth_rotation45']

import os
import shutil
n = 10
for img_dir, mask_dir in zip(img_dirs, mask_dirs):
    for id in range(1, 11):
        file_name = f'satImage_{id:03d}.png'
        
        val_image_dir = 'datasets/training/val_image'
        os.makedirs(val_image_dir, exist_ok=True)
        val_mask_dir = 'datasets/training/val_mask'
        os.makedirs(val_mask_dir, exist_ok=True)
        
        src_file = os.path.join(img_dir, file_name)
        dst_file = os.path.join(val_image_dir, f'satImage_{id+n:03d}.png')
        shutil.move(src_file, dst_file)

        src_file = os.path.join(mask_dir, file_name)
        dst_file = os.path.join(val_mask_dir, f'satImage_{id+n:03d}.png')
        shutil.move(src_file, dst_file)
        
    n += 10
