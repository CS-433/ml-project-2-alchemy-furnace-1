import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from glob import glob
import os
import math
from tqdm import tqdm
# set the path to the dataset
path = 'datasets/training/groundtruth_binary'
path_img = 'datasets/training/ori_images'

output_paths = {
    'rotation_mask': 'datasets/training/groundtruth_rotation',
    'rotation_img': 'datasets/training/image_rotation',
    'flip_mask': 'datasets/training/groundtruth_flip',
    'flip_img': 'datasets/training/image_flip'
}

# create the output directories
for out_path in output_paths.values():
    os.makedirs(out_path, exist_ok=True)

# Data Augmentation
def random_rotation(image1, image2):
    angle = random.choice([90, 180, 270])
    return image1.rotate(angle), image2.rotate(angle)

def random_flip(image1, image2):
    flip_method = random.choice([Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM])
    return image1.transpose(flip_method), image2.transpose(flip_method)


# apply the transformations
targets = glob(os.path.join(path, '*.png'))
for t in tqdm(targets):
    basename = os.path.basename(t)
    mask_image = Image.open(t).convert('L')  
    img_path = os.path.join(path_img, basename)
    if not os.path.exists(img_path):
        print(f"Image not found for mask: {basename}")
        continue
    img_image = Image.open(img_path).convert('RGB')  # ensure the image is in RGB mode

    # Random Rotation
    rotated_mask, rotated_img = random_rotation(mask_image, img_image)

    rotated_mask.save(os.path.join(output_paths['rotation_mask'], basename))
    rotated_img.save(os.path.join(output_paths['rotation_img'], basename))

    # Random Flip
    flipped_mask, flipped_img = random_flip(mask_image, img_image)
    flipped_mask.save(os.path.join(output_paths['flip_mask'], basename))
    flipped_img.save(os.path.join(output_paths['flip_img'], basename))




