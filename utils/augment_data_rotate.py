import cv2
import numpy as np
import random
import os
from tqdm import tqdm

# set the path to the dataset
gt_path = 'datasets/training/groundtruth_binary'
img_path = 'datasets/training/ori_images'

output_paths = {
    'rotation45_mask': 'datasets/training/groundtruth_rotation45',
    'rotation45_img': 'datasets/training/image_rotation45',
}

# create the output directories
for out_path in output_paths.values():
    os.makedirs(out_path, exist_ok=True)

def get_img_mask(id: int):
    id = id if id <= 100 else id % 100
    img = cv2.imread(os.path.join(img_path, f"satImage_{id:03d}.png"))
    mask = cv2.imread(os.path.join(gt_path, f"satImage_{id:03d}.png"), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        print(f'satImage_{id:03d}.png not found')
        return None, None
    return img, mask

def resize(img, mask, target_img, target_mask):
    img = cv2.resize(img, (target_img.shape[1], target_img.shape[0]))
    mask = cv2.resize(mask, (target_mask.shape[1], target_mask.shape[0]))
    return img, mask

def splice_rotation(id):
    angle = random.choice([45, 135, 225, 315])
    img5, mask5 = get_img_mask(id)
    (h, w) = img5.shape[:2]
    imgs_masks = [resize(*get_img_mask(id + i), img5, mask5) for i in range(1, 5)]
    img2, mask2 = imgs_masks[0]
    img4, mask4 = imgs_masks[1]
    img6, mask6 = imgs_masks[2]
    img8, mask8 = imgs_masks[3]
    img1, img3, img7, img9 = np.zeros_like(img5), np.zeros_like(img5), np.zeros_like(img5), np.zeros_like(img5)
    mask1, mask3, mask7, mask9 = np.zeros_like(mask5), np.zeros_like(mask5), np.zeros_like(mask5), np.zeros_like(mask5)

    top = np.concatenate([img1, img2, img3], axis=1)
    middle = np.concatenate([img4, img5, img6], axis=1)
    bottom = np.concatenate([img7, img8, img9], axis=1)
    final_img = np.concatenate([top, middle, bottom], axis=0)

    top_mask = np.concatenate([mask1, mask2, mask3], axis=1)
    middle_mask = np.concatenate([mask4, mask5, mask6], axis=1)
    bottom_mask = np.concatenate([mask7, mask8, mask9], axis=1)
    final_mask = np.concatenate([top_mask, middle_mask, bottom_mask], axis=0)

    center = (final_img.shape[1] // 2, final_img.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    final_img = cv2.warpAffine(final_img, rotation_matrix, (final_img.shape[1], final_img.shape[0]))
    final_mask = cv2.warpAffine(final_mask, rotation_matrix, (final_mask.shape[1], final_mask.shape[0]))
    final_img = final_img[h:2*h, w:2*w]
    final_mask = final_mask[h:2*h, w:2*w]
    normalized_img_array = np.where(final_mask < 128, 0, 255)
    final_mask = normalized_img_array.astype(np.uint8)
    return final_img, final_mask




for t in tqdm(range(100)):
    img, mask = splice_rotation(t+1)
    cv2.imwrite(os.path.join(output_paths['rotation45_mask'], f"satImage_{t+1:03d}.png"), mask)
    cv2.imwrite(os.path.join(output_paths['rotation45_img'], f"satImage_{t+1:03d}.png"), img)



