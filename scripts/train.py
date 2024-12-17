import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.data import ConcatDataset
import datetime
import json
import wandb
from evaluate import evaluate
from unet import UNet
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

# dir_img = Path('datasets/training/images_train')
# dir_mask = Path('datasets/training/groundtruth_binary_train')
dir_img = Path('datasets/training/ori_images')
dir_mask = Path('datasets/training/groundtruth_binary')
val_img = Path('datasets/training/val_image')
val_mask = Path('datasets/training/val_mask')
# val_img = Path('datasets/training/val_imagev2')
# val_mask = Path('datasets/training/val_maskv2')
dir_checkpoint = Path('./checkpoints/')
augmented_dir = {
    'rotation_mask': 'datasets/training/groundtruth_rotation',
    'rotation_img': 'datasets/training/image_rotation',
    'flip_mask': 'datasets/training/groundtruth_flip',
    'flip_img': 'datasets/training/image_flip',
    # 'rotation45_mask': 'datasets/training/groundtruth_rotation45',
    # 'rotation45_img': 'datasets/training/image_rotation45',
    'rotation45_maskv3': 'datasets/training/groundtruth_rotation45v3',
    'rotation45_imgv3': 'datasets/training/image_rotation45v3',
    # 'rotation45_maskv4': 'datasets/training/groundtruth_rotation45v4',
    # 'rotation45_imgv4': 'datasets/training/image_rotation45v4',
}

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 4e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-5,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):  
    #n_classes = model.module.n_classes
    
    output_dir = os.path.join(dir_checkpoint, datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    os.makedirs(output_dir, exist_ok=True)
    params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "val_percent": val_percent,
        "save_checkpoint": save_checkpoint,
        "img_scale": img_scale,
        "amp": amp,
        "weight_decay": weight_decay,
        "device": device.type,
    }
    with open(os.path.join(output_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=4)
        
    # 1. Create original dataset
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    augmented_datasets = {}
    for key, (mask_dir, img_dir) in [
        ('rotation', (augmented_dir['rotation_mask'], augmented_dir['rotation_img'])),
        ('flip', (augmented_dir['flip_mask'], augmented_dir['flip_img'])),
        # ('rotation45', (augmented_dir['rotation45_mask'], augmented_dir['rotation45_img'])),
        ('rotation45v3', (augmented_dir['rotation45_maskv3'], augmented_dir['rotation45_imgv3'])),
        # ('rotation45v4', (augmented_dir['rotation45_maskv4'], augmented_dir['rotation45_imgv4'])),
        
    ]:
        aug_dataset = BasicDataset(img_dir, mask_dir, img_scale)
        augmented_datasets[key] = aug_dataset


    combined_dataset = ConcatDataset([dataset] + list(augmented_datasets.values()))
    
    val_set = BasicDataset(val_img, val_mask, img_scale)
    train_set = combined_dataset
    n_val = len(val_set)
    n_train = len(train_set)
    
    
    
    # 5. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(model.parameters(),
    #                           lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, 
    #     T_0=10,  
    #     T_mult=3,  
    #     eta_min=1e-5  
    # )
    grad_scaler = torch.amp.GradScaler('cuda', enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
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

                        val_score = evaluate(model, val_loader, device, amp)
                        # scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass
        scheduler.step()
        if epoch % 5 == 0 or epoch == epochs:
            if save_checkpoint:
                state_dict = model.state_dict()
                state_dict['mask_values'] = dataset.mask_values
                torch.save(state_dict, os.path.join(output_dir, 'ckpt_e{}_{}.pth'.format(epoch, val_score)))
                logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--weight-decay', '-wd', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)
    

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    # model = torch.nn.DataParallel(model)
    model = model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            weight_decay=args.weight_decay
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            weight_decay=args.weight_decay
        )