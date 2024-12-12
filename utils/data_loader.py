import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch
import numpy as np
import random
import os, math

from torch.utils.data import DataLoader, Dataset

def load_image(path, gt=False):

    if gt:
        image = Image.open(path).convert('L')
    else:
        image = Image.open(path).convert('RGB')
        
    transform = transforms.ToTensor()

    image = transform(image)
        
    return image

def compress_image(img, patch_size=16, thres=0.25):
    w = img.shape[1]
    h = img.shape[2]
    c = img.shape[0]

    compressed = torch.zeros(c, int(w / patch_size), int(h / patch_size))
    
    for i in range(0, compressed.shape[1]):
        for j in range(0, compressed.shape[2]):
            patch = img[:, patch_size * i : patch_size * (i + 1), patch_size * j : patch_size * (j + 1)]
            patch_mean = patch.mean(dim=(1, 2))
            compressed[:, i, j] = (patch_mean > thres)

    return compressed 
    

def view_image(img):
    img = transforms.functional.to_pil_image(img)
    img = np.array(img)

    plt.imshow(img, cmap='Greys_r')
    plt.show()


class RandomHFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, gt_image = sample['image'], sample['gt_image']
        
        if random.random() < self.prob:
            image = F.hflip(image)
            gt_image = F.hflip(gt_image)

        return {'image': image, 'gt_image': gt_image}
    
class RandomRotate(object): 
    def __call__(self, sample):
        image, gt_image = sample['image'], sample['gt_image']

        num_rotation = random.randint(0, 3)
        image = F.rotate(image, num_rotation * 90)
        gt_image = F.rotate(gt_image, num_rotation * 90)

        return {'image': image, 'gt_image': gt_image}
    
class Rotate45(object):
    def __call__(self, sample):
        image, gt_image = sample['image'], sample['gt_image']

        C = image.size(0)
        H = image.size(1)
        W = image.size(2)

        H1 = int(H / 2 - math.sqrt(2) * H / 4)
        H2 = int(H / 2 + math.sqrt(2) * H / 4)
        W1 = int(W / 2 - math.sqrt(2) * W / 4)
        W2 = int(W / 2 + math.sqrt(2) * W / 4)

        image_rotated = F.rotate(image, 45)
        gt_image_rotated = F.rotate(gt_image, 45)

        image_resized = torch.empty(C, H2 - H1, W2 - W1)
        gt_image_resized = torch.empty(1, H2 - H1, W2 - W1)

        image_resized[:] = image_rotated[:, H1:H2, W1:W2]
        gt_image_resized[:] = gt_image_rotated[:, H1:H2, W1:W2]

        image_resized = F.resize(image_resized, size=(H, W))
        gt_image_resized = F.resize(gt_image_resized, size=(H, W))

        return {'image': image_resized, 'gt_image': gt_image_resized}

class Gaussian_Noise(object):
    def __init__(self, std=0.1):
        self.std = std
    def __call__(self, sample):
        image, gt_image = sample['image'], sample['gt_image']  
        noise = torch.randn_like(image) * self.std
        image += noise
        image = torch.clamp(image, 0, 1)

        return {'image': image, 'gt_image': gt_image}
        


        

class RoadSegmentationDataset(Dataset):
    def __init__(self, root_dir, files, transform=None):
        self.root_dir = root_dir
        self.image_dir = self.root_dir + "images/"
        self.gt_dir = self.root_dir + "groundtruth/"
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_paths = [self.image_dir + f for f in self.files]
        gt_paths = [self.gt_dir + f for f in self.files]
        
        image = load_image(image_paths[idx])
        gt_image = load_image(gt_paths[idx], gt=True)

        
        sample = {
            'image':image,
            'gt_image':gt_image
        }
        
        if self.transform:
            sample = self.transform(sample)

        sample = {
            'image':sample['image'],
            'gt_image':compress_image(sample['gt_image'])
        }

        return sample

class RoadSegmentationDataset_Augmented(Dataset):
    def __init__(self, root_dir, files, transform=None):
        self.root_dir = root_dir
        self.image_dir = self.root_dir + "images/"
        self.gt_dir = self.root_dir + "groundtruth/"
        self.files = files
        self.augmentation = transforms.Compose([RandomHFlip(prob=0.5), RandomRotate()])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        original_idx = idx // 2
        is_augmented = idx % 2

        image_paths = [self.image_dir + f for f in self.files]
        gt_paths = [self.gt_dir + f for f in self.files]
        
        image = load_image(image_paths[original_idx])
        gt_image = load_image(gt_paths[original_idx], gt=True)
                        
        sample = {
            'image':image,
            'gt_image':gt_image
        }

        if is_augmented:
            sample = self.augmentation(sample)   
                 
        if self.transform:
            sample = self.transform(sample)

        sample = {
            'image':sample['image'],
            'gt_image':compress_image(sample['gt_image'])
        }

        return sample
    
class RoadSegmentationDataset_Rotated(Dataset):
    def __init__(self, root_dir, files, transform=None):
        self.root_dir = root_dir
        self.image_dir = self.root_dir + "images/"
        self.gt_dir = self.root_dir + "groundtruth/"
        self.files = files
        self.augmentation = transforms.Compose([RandomHFlip(prob=0.5), RandomRotate()])
        self.rotation = Rotate45()
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        original_idx = idx // 3
        augmentation_idx = idx % 3

        image_paths = [self.image_dir + f for f in self.files]
        gt_paths = [self.gt_dir + f for f in self.files]
        
        image = load_image(image_paths[original_idx])
        gt_image = load_image(gt_paths[original_idx], gt=True)
            
        
        sample = {
            'image':image,
            'gt_image':gt_image
        }

        if augmentation_idx == 1:
            sample = self.augmentation(sample)  

        if augmentation_idx == 2:
            if random.random() < 0.34:
                sample = self.rotation(sample) 
            else:
                sample = self.augmentation(sample)
            
                 
        if self.transform:
            sample = self.transform(sample)
        
        sample = {
            'image':sample['image'],
            'gt_image':compress_image(sample['gt_image'])
        }

        return sample

class TestSet(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        files = os.listdir(self.root_dir)
        return len(files)
    
    def __getitem__(self, idx):
        files = os.listdir(self.root_dir)
        image_paths = [f"{self.root_dir}test_{i+1}/test_{i+1}.png" for i in range(len(files))]

        image = load_image(image_paths[idx])

        sample = {
            'image': image,
            'idx': idx
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

class RoadSegmentationDataset_NonDuplicated(Dataset):
    def __init__(self, root_dir, files, transform=None):
        self.root_dir = root_dir
        self.image_dir = self.root_dir + "images/"
        self.gt_dir = self.root_dir + "groundtruth/"
        self.files = files
        self.RFlip = RandomHFlip()
        self.RRotate = RandomRotate()
        self.Rotate45 = Rotate45()
        self.Noise = Gaussian_Noise()
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_paths = [self.image_dir + f for f in self.files]
        gt_paths = [self.gt_dir + f for f in self.files]
        
        image = load_image(image_paths[idx])
        gt_image = load_image(gt_paths[idx], gt=True)

        
        sample = {
            'image':image,
            'gt_image':gt_image
        }

        sample = self.RFlip(sample)

        if random.random() < 0.5:
            sample = self.RRotate(sample)

        if random.random() < 0.1:
            sample = self.Rotate45(sample)

        sample = self.Noise(sample)
        
        if self.transform:
            sample = self.transform(sample)

        sample = {
            'image':sample['image'],
            'gt_image':compress_image(sample['gt_image'])
        }

        return sample