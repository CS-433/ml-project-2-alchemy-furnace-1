import numpy as np
from PIL import Image
import random
from glob import glob
import os
import math
from tqdm import tqdm

# set the path to the dataset
path = 'datasets/training/groundtruth_binary'
path_img = 'datasets/training/ori_images'

output_paths = {
    'rotation_mask': 'datasets/training/groundtruth_rotation45v3',
    'rotation_img': 'datasets/training/image_rotation45v3',
}

for out_path in output_paths.values():
    os.makedirs(out_path, exist_ok=True)

angles = [45, 135, 225, 315]

targets = glob(os.path.join(path, '*.png'))
for t in tqdm(targets):
    basename = os.path.basename(t)
    mask_image = Image.open(t).convert('L') 
    img_path = os.path.join(path_img, basename)
    if not os.path.exists(img_path):
        print(f"Image not found for mask: {basename}")
        continue
    img_image = Image.open(img_path).convert('RGB')  
#  random choice of angle
    angle = random.choice(angles)
    angle += random.uniform(-5, 5)  
    

    W, H = img_image.size
    # calculate the new image size
    H1 = int(H / 2 - math.sqrt(2) * H / 4)
    H2 = int(H / 2 + math.sqrt(2) * H / 4)
    W1 = int(W / 2 - math.sqrt(2) * W / 4)
    W2 = int(W / 2 + math.sqrt(2) * W / 4)
    # rotate the image
    rotated_mask = mask_image.rotate(angle)
    rotated_img = img_image.rotate(angle)

    # crop the image and resize it to the original size
    rotated_mask = rotated_mask.crop((W1, H1, W2, H2)).resize((W, H), Image.NEAREST)
    rotated_img = rotated_img.crop((W1, H1, W2, H2)).resize((W, H), Image.BILINEAR)

    # save the image
    base_name, ext = os.path.splitext(basename)
    new_basename = f"{base_name}{ext}"

    rotated_mask.save(os.path.join(output_paths['rotation_mask'], new_basename))
    rotated_img.save(os.path.join(output_paths['rotation_img'], new_basename))
