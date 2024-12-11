import argparse
import sys
import logging
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import wandb
from tqdm import tqdm

from utils.data_loader import view_image, load_image
from utils.data_loader import RoadSegmentationDataset, RoadSegmentationDataset_Augmented, RoadSegmentationDataset_Rotated
from utils.data_loader import RandomRotate, RandomHFlip
from utils.dice_score import dice_loss, f1_loss
from unet_mini import UNetMini, UNetMiniPro
from evaluate import evaluate

root_dir = './training/'
checkpoint_dir = Path('./checkpoints/')


def train(
        model, 
        device, 
        epochs: int = 5, 
        batch_size: int = 1, 
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        amp: bool = False,
        weight_decay: float = 1e-5,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        thres: float = 0.25
):
    

    # 1. Split into train / validation partitions
    root_dir = './training/'
    image_dir = root_dir + "images/"
    gt_dir = root_dir + "groundtruth/"
    files = os.listdir(image_dir)

    train_files = random.sample(files, int(len(files) * (1 - val_percent)))
    val_files = list(set(files) - set(train_files))
    n_train = len(train_files)
    n_val = len(val_files)

    # 2. Create the train set and the validation set
    tsfm = transforms.Compose([RandomHFlip(prob=0.5), RandomRotate()])
    train_set = RoadSegmentationDataset_Rotated(root_dir, train_files, transform=None)
    val_set = RoadSegmentationDataset(root_dir, val_files)

    # 3. Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net_Mini', resume='allow')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)

    grad_scaler = torch.amp.GradScaler('cuda', enabled=amp)
    criterion = nn.BCEWithLogitsLoss()
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit='img') as pbar:
            for batch in train_loader:

                batch_x, batch_y = batch['image'], batch['gt_image']
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                with torch.autocast(device.type, enabled=amp):
                    y_pred = model(batch_x)
                    loss = criterion(y_pred.squeeze(1), batch_y.squeeze(1))
                    loss += f1_loss(F.sigmoid(y_pred.squeeze(1)), batch_y.squeeze(1), thres=thres)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(batch_x.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evalutation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp, thres)
                        
                        logging.info('Validation Dice / F1 Score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation dice / f1': val_score,
                                'images': wandb.Image(batch_x[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(batch_y[0].float().cpu()),
                                    'pred': wandb.Image(y_pred[0].float().cpu()),
                                    'classified': wandb.Image((y_pred[0]>thres).float().cpu()),
                                },
                                'thres': thres,
                                'batch_size': batch_size,
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        scheduler.step(val_score)
        if save_checkpoint:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(checkpoint_dir / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f"Checkpoint{epoch} saved!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_seg = UNetMini(n_channels=3).to(device)
epochs = 100
batch_size = 8
learning_rate = 2e-4
val_percent = 0.2
save_checkpoint = True
amp = True
weight_decay = 1e-5
momentum = 0.999
gradient_clipping = 1.0
thres = 0.3

model_seg = UNetMiniPro(n_channels=3).to(device)
train(model_seg, device, epochs, batch_size, learning_rate, val_percent, save_checkpoint, amp, weight_decay, momentum, gradient_clipping, thres)

root_dir = './training/'
image_dir = root_dir + "images/"
gt_dir = root_dir + "groundtruth/"
files = os.listdir(image_dir)

train_files = random.sample(files, 80)
val_files = list(set(files) - set(train_files))

print("Loading training images")
train_imgs = [load_image(image_dir + f) for f in train_files]
train_gt_imgs = [load_image(gt_dir + f, gt=True) for f in train_files]

print("Loading validation images")
val_imgs = [load_image(image_dir + f) for f in val_files]
val_gt_imgs = [load_image(gt_dir + f, gt=True) for f in val_files]


model_seg.to('cuda')
model_seg.eval()

train_img = train_imgs[0].unsqueeze(0).to('cuda')

tensor = val_imgs[0].unsqueeze(0).to('cuda')
pred = model_seg(tensor)

pred_img = (pred.squeeze(0) > thres).float()
view_image(val_imgs[0])
view_image(pred_img)
view_image(val_gt_imgs[0])
