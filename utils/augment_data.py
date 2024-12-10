import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from glob import glob
import os
import math
from tqdm import tqdm
# 路径设置
path = 'datasets/training/groundtruth_binary'
path_img = 'datasets/training/images'

output_paths = {
    'rotation_mask': 'datasets/training/groundtruth_rotation',
    'rotation_img': 'datasets/training/image_rotation',
    'flip_mask': 'datasets/training/groundtruth_flip',
    'flip_img': 'datasets/training/image_flip'
    # 'crop_mask': 'datasets/training/groundtruth_crop',
    # 'crop_img': 'datasets/training/image_crop',
    # 'color_mask': 'datasets/training/groundtruth_color',
    # 'color_img': 'datasets/training/image_color',
}

# 创建输出目录
for out_path in output_paths.values():
    os.makedirs(out_path, exist_ok=True)

# 数据增强函数
def random_rotation(image1, image2):
    angle = random.choice([90, 180, 270])
    return image1.rotate(angle), image2.rotate(angle)

def random_flip(image1, image2):
    flip_method = random.choice([Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM])
    return image1.transpose(flip_method), image2.transpose(flip_method)

def random_crop(image1, image2, crop_size=(256, 256)):
    import cv2
    import numpy as np

    image1_np = np.array(image1)
    image2_np = np.array(image2)
    height, width = image1_np.shape[:2]

    if width < crop_size[0] or height < crop_size[1]:
        crop_size = (min(width, crop_size[0]), min(height, crop_size[1]))

    left = random.randint(0, width - crop_size[0]) if width > crop_size[0] else 0
    top = random.randint(0, height - crop_size[1]) if height > crop_size[1] else 0
    right = left + crop_size[0]
    bottom = top + crop_size[1]

    cropped_image1_np = image1_np[top:bottom, left:right]
    cropped_image2_np = image2_np[top:bottom, left:right]

    resized_image1_np = cv2.resize(cropped_image1_np, (width, height), interpolation=cv2.INTER_LANCZOS4)
    resized_image2_np = cv2.resize(cropped_image2_np, (width, height), interpolation=cv2.INTER_LANCZOS4)

    _, resized_image1_np = cv2.threshold(resized_image1_np, 127, 255, cv2.THRESH_BINARY)

    resized_image1 = Image.fromarray(resized_image1_np)
    resized_image2 = Image.fromarray(resized_image2_np)

    return resized_image1, resized_image2

def color_jitter(image):
    enhancers = [
        ImageEnhance.Brightness(image),
        ImageEnhance.Contrast(image),
        ImageEnhance.Color(image),
        ImageEnhance.Sharpness(image)
    ]
    for enhancer in enhancers:
        factor = random.uniform(1.5, 2.0)  # 调整因子范围更保守
        image = enhancer.enhance(factor)
    return image




# 遍历并增强数据
targets = glob(os.path.join(path, '*.png'))
for t in tqdm(targets):
    basename = os.path.basename(t)
    mask_image = Image.open(t).convert('L')  # 确保掩码是单通道
    img_path = os.path.join(path_img, basename)
    if not os.path.exists(img_path):
        print(f"Image not found for mask: {basename}")
        continue
    img_image = Image.open(img_path).convert('RGB')  # 确保图像是RGB

    # Random Rotation
    rotated_mask, rotated_img = random_rotation(mask_image, img_image)

    rotated_mask.save(os.path.join(output_paths['rotation_mask'], basename))
    rotated_img.save(os.path.join(output_paths['rotation_img'], basename))

    # Random Flip
    flipped_mask, flipped_img = random_flip(mask_image, img_image)
    flipped_mask.save(os.path.join(output_paths['flip_mask'], basename))
    flipped_img.save(os.path.join(output_paths['flip_img'], basename))

    # # Random Crop
    # cropped_mask, cropped_img = random_crop(mask_image, img_image)
    # cropped_mask.save(os.path.join(output_paths['crop_mask'], basename))
    # cropped_img.save(os.path.join(output_paths['crop_img'], basename))

    # # # Color Jitter (仅对图像进行，掩码不变)
    # jittered_img = color_jitter(img_image)
    # jittered_img.save(os.path.join(output_paths['color_img'], basename))
    # jittered_mask = mask_image
    # jittered_mask.save(os.path.join(output_paths['color_mask'], basename))



print("数据增强完成。")
