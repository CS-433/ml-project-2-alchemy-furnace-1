import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class RoadSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (400, 400))  # Resize to 400x400
        mask = cv2.resize(mask, (400, 400))   # Resize to 400x400

        # Normalize image to [0, 1]
        image = image / 255.0
        mask = np.expand_dims(mask, axis=-1)  # Ensure mask has shape [height, width, 1]
        mask = mask / 255.0  # Normalize mask to [0, 1]

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), torch.tensor(mask, dtype=torch.float32)

def prepare_data(image_dir, mask_dir, batch_size=16):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = RoadSegmentationDataset(image_dir, mask_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
