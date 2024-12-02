import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import matplotlib.image as mpimg
import numpy
import sys
IMG_PATCH_SIZE = 16
def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = torch.sum(v)  # 使用 PyTorch 的 sum 方法来保留计算图

    label = torch.where(df > foreground_threshold,
                        torch.tensor([0.0], device=v.device, dtype=torch.float32),
                        torch.tensor([1.0], device=v.device, dtype=torch.float32))
    return label
def value_to_class2(v):
    df = torch.sum(v)  # 使用 PyTorch 的 sum 方法来保留计算图
    return df
def img_crop(im, w, h):
    im_patchs=[]
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            im_patchs.append(im_patch)
    return im_patchs

def extract_labels_torch2(gt_img, image_PZ=16):
    """
    Extract the labels into a 1-hot matrix [image index, label index] using PyTorch operations.

    Args:
    - gt_img (Tensor): Ground truth image, expects a 2D or 3D tensor.
    - image_PZ (int): Patch size, the size of each patch to extract.

    Returns:
    - Tensor: Extracted labels as a tensor.
    """
    if not isinstance(gt_img, torch.Tensor):
        raise ValueError("Expected gt_img to be a torch.Tensor")

    img_width, img_height = gt_img.shape[-2:]  # 获取高度和宽度

    labels = []
    for i in range(0, img_height, image_PZ):
        for j in range(0, img_width, image_PZ):
            patch = gt_img[j:j + image_PZ, i:i + image_PZ]
            patch_mean = patch.mean()  # 保留计算图

            label = value_to_class(patch_mean)  # 假设它返回一个保留计算图的张量
            labels.append(label)

    # 将所有标签拼接成一个张量
    labels_tensor = torch.stack(labels) ##if len(labels) > 1 else labels[0]
    return labels_tensor
import torch

def extract_labels_torch(gt_img, image_PZ=16):
    """
    Extract the labels into a 1-hot matrix [image index, label index] using PyTorch operations.

    Args:
    - gt_img (Tensor): Ground truth image, expects a 2D or 3D tensor.
    - image_PZ (int): Patch size, the size of each patch to extract.

    Returns:
    - Tensor: Extracted labels as a tensor.
    """
    if not isinstance(gt_img, torch.Tensor):
        raise ValueError("Expected gt_img to be a torch.Tensor")

    img_width, img_height = gt_img.shape[-2:]  # 获取高度和宽度

    labels = []
    for i in range(0, img_height, image_PZ):
        for j in range(0, img_width, image_PZ):
            patch = gt_img[j:j + image_PZ, i:i + image_PZ]
            patch_mean = patch.mean()  # 保留计算图

            # label = value_to_class2(patch_mean)  # 假设它返回一个保留计算图的张量
            labels.append(patch_mean)

    # 将所有标签拼接成一个张量
    labels_tensor = torch.stack(labels) ##if len(labels) > 1 else labels[0]
    return labels_tensor

class RoadSegmentationDataset(Dataset):
    def __init__(self, groundtruth_dir, images_dir, operations, transform=None):
        self.image_paths = []
        self.label_paths = []
        self.transform = transform

        for op in operations:
            gt_dir = f"{groundtruth_dir}_{op}"
            img_dir = f"{images_dir}_{op}"
            
            for filename in os.listdir(gt_dir):
                if filename.endswith('.png') or filename.endswith('.jpg'):
                    gt_path = os.path.join(gt_dir, filename)
                    img_path = os.path.join(img_dir, filename)
                    
                    if os.path.exists(img_path):
                        self.label_paths.append(gt_path)
                        self.image_paths.append(img_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = Image.open(self.label_paths[idx]).convert('L')
        
        if self.transform:
            image = self.transform(image)
            label = transforms.ToTensor()(label)
        # print('image.shape',image.shape)
        # print('label', label.shape)
        # label = torch.where(label > 0, 1.0, 0.0).type(torch.float32)
        # label = extract_labels(label)
        # print('label',label.shape,file=sys.stdout, flush=True)
        return image, label

# define the input directories and operations
# base_input_dirs = ['training/groundtruth', 'training/images']
# operations = ['flip1', 'flip2', 'orgin', 'rotate_90', 'rotate_180', 'rotate_270']

# # create the output directories
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# dataset = RoadSegmentationDataset('groundtruth', 'images', operations, transform=transform)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# print("DataLoader is ready to use.")
