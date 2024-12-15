import numpy as np
from PIL import Image
import random
from glob import glob
import os
import math
from tqdm import tqdm

# 路径设置
path = 'datasets/training/groundtruth_binary'
path_img = 'datasets/training/ori_images'

output_paths = {
    'rotation_mask': 'datasets/training/groundtruth_rotation45v3',
    'rotation_img': 'datasets/training/image_rotation45v3',
}

# 创建输出目录
for out_path in output_paths.values():
    os.makedirs(out_path, exist_ok=True)

# 要随机选择的旋转角度列表
angles = [45, 135, 225, 315]

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

    # 随机选择一个旋转角度
    angle = random.choice(angles)

    W, H = img_image.size
    # 计算旋转后需要裁剪的区域(原代码中已有逻辑)
    H1 = int(H / 2 - math.sqrt(2) * H / 4)
    H2 = int(H / 2 + math.sqrt(2) * H / 4)
    W1 = int(W / 2 - math.sqrt(2) * W / 4)
    W2 = int(W / 2 + math.sqrt(2) * W / 4)

    # 对mask和image进行旋转
    rotated_mask = mask_image.rotate(angle)
    rotated_img = img_image.rotate(angle)

    # 裁剪并resize回原始尺寸
    rotated_mask = rotated_mask.crop((W1, H1, W2, H2)).resize((W, H), Image.NEAREST)
    rotated_img = rotated_img.crop((W1, H1, W2, H2)).resize((W, H), Image.BILINEAR)

    # 保存新文件
    base_name, ext = os.path.splitext(basename)
    new_basename = f"{base_name}{ext}"

    rotated_mask.save(os.path.join(output_paths['rotation_mask'], new_basename))
    rotated_img.save(os.path.join(output_paths['rotation_img'], new_basename))
