import argparse
import logging
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
from scripts.mask_to_submission import masks_to_submission
def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=False)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(model_path, input_files):
    model_name = os.path.basename(model_path).replace('.pth', '')
    output_dir = os.path.join('outputs', model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_files = []
    for input_file in input_files:
        output_file = os.path.join(output_dir, os.path.basename(input_file).replace('.png', '_OUT.png'))
        output_files.append(output_file)
    
    return output_files


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    args.model = '/home/yifwang/ml-project-2-alchemy-furnace-1/checkpoints/checkpoint_epoch80_0.9015659689903259.pth'
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    input_path = 'datasets/test_set_images'
    input_images = glob.glob(os.path.join(input_path, '*/*.png'))
    
    in_files = input_images #datasets/test_set_images/test_1/test_1.png
    out_files = get_output_filenames(args.model, in_files) #outputs/checkpoint_epoch20_0.7186881899833679/test_1_OUT.png

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        # img_flip1 = img.transpose(Image.FLIP_TOP_BOTTOM)
        # img_flip2 = img.transpose(Image.FLIP_LEFT_RIGHT)
        # img_rot90 = img.rotate(90)
        # img_rot180 = img.rotate(180)
        # img_rot270 = img.rotate(270)
        # mask_flip1 = predict_img(net=net,
        #                          full_img=img_flip1,
        #                          scale_factor=args.scale,
        #                          out_threshold=args.mask_threshold,
        #                          device=device)
        # mask_flip2 = predict_img(net=net,
        #                             full_img=img_flip2,
        #                             scale_factor=args.scale,
        #                             out_threshold=args.mask_threshold,
        #                             device=device)
        # mask_rot90 = predict_img(net=net,
        #                             full_img=img_rot90,
        #                             scale_factor=args.scale,
        #                             out_threshold=args.mask_threshold,
        #                             device=device)
        # mask_rot180 = predict_img(net=net,
        #                             full_img=img_rot180,
        #                             scale_factor=args.scale,
        #                             out_threshold=args.mask_threshold,
        #                             device=device)
        # mask_rot270 = predict_img(net=net,
        #                           full_img=img_rot270,
        #                             scale_factor=args.scale,
        #                             out_threshold=args.mask_threshold,
        #                             device=device)
        # mask_flip1 = np.flipud(mask_flip1)
        # mask_flip2 = np.fliplr(mask_flip2)
        # mask_rot90 = np.rot90(mask_rot90, 3)
        # mask_rot180 = np.rot90(mask_rot180, 2)
        # mask_rot270 = np.rot90(mask_rot270, 1)
        # mask = mask + mask_flip1 + mask_flip2 + mask_rot90 + mask_rot180 + mask_rot270

        # mask = mask / 6.0
        # mask = mask + mask_flip1 + mask_flip2
        # mask = mask / 3.0

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')
            
    submission_filename = f'submission_{args.model.split("/")[-1].split(".pth")[0]}.csv'
    print(submission_filename)
    image_filenames = []
    files_path = out_files[0].rsplit('/', 1)[0]
    for i in range(1, 51):
        image_filename = os.path.join(files_path, 'test_' + str(i) + '_OUT.png')
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)