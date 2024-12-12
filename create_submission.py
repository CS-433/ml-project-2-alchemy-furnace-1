import os

import torch

from unet_mini import UNetMini, UNetMiniPro

from utils.data_loader import TestSet
from torch.utils.data import DataLoader


def load_model(
        checkpoint_path='./checkpoints/',
        epoch=100,
        ):
    model_path = f"{checkpoint_path}checkpoint_epoch{epoch}.pth"
    state_dict = torch.load(model_path, weights_only=False)
    model = UNetMiniPro(n_channels=3)
    model.load_state_dict(state_dict)
    model = model.to('cuda')

    return model

def create_submission(
        model,
        root_dir = './test_set_images/',
        submission_filename = 'submission.csv',
        thres = 0.25
        ):
    testset = TestSet(root_dir)
    dataset = DataLoader(testset, batch_size=1, shuffle=False)

    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for batch in dataset:
            img, idx = batch['image'], batch['idx']
            img = img.to('cuda')
            idx = idx.to('cuda')
            model.eval()
            pred = model(img)
            mask = (pred.squeeze((0, 1)) > thres)
            for j in range(0, mask.shape[1]):
                for i in range(0, mask.shape[0]):
                    label = int(mask[i, j])
                    f.writelines("{:03d}_{}_{},{}\n".format(int(idx)+1, 16 * j, 16 * i, label))

root_dir = './test_set_images/'
checkpoint_path = './checkpoints/'
submission_filename = 'submission.csv'
epoch = 100
thres = 0.3

model = load_model(checkpoint_path=checkpoint_path, epoch=epoch)
create_submission(model=model, root_dir=root_dir, submission_filename=submission_filename, thres=thres)