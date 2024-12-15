import os
import math
from PIL import Image
import torch
import torchvision.transforms.functional as F

def augment_images_with_rotation(input_dir, operations):
    for filename in os.listdir(input_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            filepath = os.path.join(input_dir, filename)
            image = Image.open(filepath)
            W, H = image.size  # Correctly unpack the tuple returned by image.size
            H1 = int(H / 2 - math.sqrt(2) * H / 4)
            H2 = int(H / 2 + math.sqrt(2) * H / 4)
            W1 = int(W / 2 - math.sqrt(2) * W / 4)
            W2 = int(W / 2 + math.sqrt(2) * W / 4)

            base_name, ext = os.path.splitext(filename)

            for op in operations:
                if op == 'rotate_45':
                    augmented_image = image.rotate(45)
                elif op == 'rotate_135':
                    augmented_image = image.rotate(135)
                elif op == 'rotate_225':
                    augmented_image = image.rotate(225)
                elif op == 'rotate_315':
                    augmented_image = image.rotate(315)
                else:
                    continue
                augmented_image = augmented_image.crop((W1, H1, W2, H2))
                augmented_image = augmented_image.resize((W, H), Image.BILINEAR)
                new_filename = f"{base_name}_{op}{ext}"
                output_path = os.path.join('./training/images', new_filename)
                augmented_image.save(output_path)

class Rotate45:
    def __call__(self, sample):
        image, gt_image = sample['image'], sample['gt_image']

        C, H, W = image.size()
        H1 = int(H / 2 - math.sqrt(2) * H / 4)
        H2 = int(H / 2 + math.sqrt(2) * H / 4)
        W1 = int(W / 2 - math.sqrt(2) * W / 4)
        W2 = int(W / 2 + math.sqrt(2) * W / 4)

        image_rotated = F.rotate(image, 45)
        gt_image_rotated = F.rotate(gt_image, 45)

        image_resized = image_rotated[:, H1:H2, W1:W2]
        gt_image_resized = gt_image_rotated[:, H1:H2, W1:W2]

        image_resized = F.resize(image_resized, size=(H, W))
        gt_image_resized = F.resize(gt_image_resized, size=(H, W))

        return {'image': image_resized, 'gt_image': gt_image_resized}

# Define input directories and operations
base_input_dirs =  ['./training/images_origin']
operations = ['rotate_45'] ##, 'rotate_135', 'rotate_225', 'rotate_315']

# Perform data augmentation for 45, 135, 225, and 315-degree rotations
for base in base_input_dirs:
    augment_images_with_rotation(base, operations)

print("45, 135, 225, and 315-degree rotations completed.")
