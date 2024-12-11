import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import dice_loss, f1_loss

@torch.inference_mode()
def evaluate(net, dataloader, device, amp, thres=0.25):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    with torch.autocast(device.type, enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            
            batch_x, batch_y = batch['image'], batch['gt_image']
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = net(batch_x)

            y_pred = (F.sigmoid(y_pred) > thres).float()
            dice_score += 1 - f1_loss(y_pred, batch_y, reduce_batch_first=False, thres=thres)

    net.train()
    return dice_score / max(num_val_batches, 1)