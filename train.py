from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torchvision import transforms
import torch
from src.enc_dec import ViTEncoder, Decoder
from datapreprocess.dataloader import RoadSegmentationDataset,img_crop,value_to_class,extract_labels_torch2,extract_labels_torch

from torchvision.models import vit_b_16
from src.enc_dec import ViTEncoderDecoder
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import sys
from src.eval import predict_labels, accuracy_score_tensors, f1_score_tensors
BATCH_SIZE = 8
PATCH_SIZE = 16
class RoadSegmentationModel(nn.Module):
    def __init__(self):
        super(RoadSegmentationModel, self).__init__()
        self.encoder = ViTEncoder()
        self.decoder = Decoder(input_dim=768, output_dim=1)  # ViT output dimension is 768

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # Smooth term to prevent division by zero

    def forward(self, outputs, targets):
        # Flatten the outputs and targets into 1D tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # Calculate the intersection between outputs and targets
        intersection = (outputs * targets).sum()
        
        # Compute the Dice coefficient
        dice_coeff = (2. * intersection + self.smooth) / (outputs.sum() + targets.sum() + self.smooth)
        
        # Calculate Dice Loss as 1 - Dice coefficient
        dice_loss = 1 - dice_coeff

        return dice_loss

if __name__ == "__main__":
    # Define input directories and operations
    base_input_dirs = ['training/groundtruth', 'training/images']
    operations = ['flip1', 'flip2', 'origin', 'rotate_90', 'rotate_180', 'rotate_270']

    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = RoadSegmentationDataset('./training/groundtruth', './training/images', operations, transform=transform)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # print('val_loader:', val_loader)
    print ("DataLoader is ready to use.", file=sys.stdout, flush=True)
    print (len(train_loader))
    print (len(val_loader))

    # Initialize model, loss, and optimizer
    model = ViTEncoderDecoder(patch_size=16)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4, betas=(0.9, 0.99))


    # Training loop
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        total_accuracy = 0.0
        total_f1 = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs.squeeze()
            # labels = labels.squeeze()
            # outputs2 = [extract_labels_torch2(output.squeeze(), image_PZ=4) for output in outputs]

            labels2 = [extract_labels_torch2(label, image_PZ=16) for label in labels]

            # outputs = torch.stack(outputs2).squeeze()
            labels = torch.stack(labels2).squeeze()
            # print('outputs',outputs.shape,file=sys.stdout, flush=True)
            # print('outputs',outputs.shape,file=sys.stdout, flush=True)
            # outputs = extract_labels_torch2(outputs.squeeze(), image_PZ=4)
            # labels = extract_labels_torch2(labels.squeeze(), image_PZ=4)
            
            # outputs = outputs.to(labels.device)
            # loss = F.mse_loss(outputs,labels)
            dice_loss_fn = DiceLoss()
            loss = dice_loss_fn(outputs, labels)
            # print('loss',loss,file=sys.stdout, flush=True)
            # recon_loss = F.mse_loss(outputs, labels)
            # print('recon_loss',recon_loss,file=sys.stdout, flush=True)
            # bce_loss = dice_loss_fn(torch.sigmoid(outputs), labels)
            # loss = recon_loss + loss
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('outputs',outputs.shape,file=sys.stdout, flush=True)
            # print('outputs',outputs.shape,file=sys.stdout, flush=True)
            # outputs2 = [extract_labels_torch2(output.squeeze(), image_PZ=16) for output in outputs]
            # labels2 = [extract_labels_torch2(label, image_PZ=16) for label in labels]

            # outputs = torch.stack(outputs2).squeeze()
            # labels = torch.stack(labels2).squeeze()
            running_loss += loss.item()
            preds = predict_labels(outputs)
            true_labels = labels
            # print('outputs',preds.shape,file=sys.stdout, flush=True)
            # print('labels',true_labels.shape,file=sys.stdout, flush=True)
            total_accuracy += accuracy_score_tensors(true_labels, preds)
            total_f1 += f1_score_tensors(true_labels, preds)
        total_accuracy/=len(train_loader)
        total_f1/=len(train_loader)

        
        scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {total_accuracy:.4f}, F1 Score: {total_f1:.4f}", file=sys.stdout, flush=True)


        # Validation step
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_f1 = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device, dtype=torch.float32)
                outputs = model(images)
                outputs2=[]
                for output in outputs:
                    output = output.squeeze()
                    output2 = extract_labels_torch(output,image_PZ=16)
                    outputs2.append(output2.clone())

                labels2=[]
                for label in labels:
                    label =label.squeeze()
                    label = extract_labels_torch2(label,image_PZ=16)
                    labels2.append(label.clone())

                outputs = torch.stack(outputs2)
                labels = torch.stack(labels2)
                outputs.to(labels.device)
                outputs = torch.stack(outputs2)
                outputs.to(labels.device)
                outputs = torch.stack(outputs2)
                outputs = outputs.squeeze()
                labels = torch.stack(labels2)
                labels = labels.squeeze()
                # loss = criterion(outputs, labels)
                # val_loss += loss.item()

                preds = predict_labels(outputs)
                true_labels = predict_labels(labels)

                val_accuracy += accuracy_score_tensors(true_labels, preds)
                val_f1 += f1_score_tensors(true_labels, preds)

            # val_loss /= len(val_loader)
            val_accuracy /= len(val_loader)
            val_f1 /= len(val_loader)
            print(f"Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}", file=sys.stdout, flush=True)

print("Training completed.")