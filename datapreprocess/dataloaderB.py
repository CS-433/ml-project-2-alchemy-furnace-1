import os
import sys
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import numpy as np

class RoadSegmentationTest(Dataset):
    def __init__(self, images_dir, transform=None):
        self.image_paths = []
        self.label_paths = []
        self.transform = transform
            
        for filename in os.listdir(images_dir):
            
            img_path = os.path.join(images_dir, filename, filename + '.png')
                
            if os.path.exists(img_path):
                self.image_paths.append(img_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        # label = Image.open(self.label_paths[idx]).convert('L')
        
        if self.transform:
            image = self.transform(image)
            # label = transforms.ToTensor()(label)
        print('image.shape',image.shape)
        # print('label', label.shape)
        # label = torch.where(label > 0, 1.0, 0.0).type(torch.float32)
        # label = extract_labels(label)
        # print('label',label.shape,file=sys.stdout, flush=True)
        return image