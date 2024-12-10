""" 
given ground truth mask is continuous, from 0,1,2... to 255
we need to convert it to binary mask, 0 and 255
"""
import numpy as np
from PIL import Image
from os.path import splitext
from glob import glob
import os
path = 'datasets/training/groundtruth'
path2 = 'datasets/training/groundtruth_binary'
os.path.exists(path2) or os.makedirs(path2)
targets = glob(path + '/*.png')
for t in targets:
    img = Image.open(t)
    img_array = np.array(img)
    normalized_img_array = np.where(img_array < 128, 0, 255)
    normalized_img = Image.fromarray(normalized_img_array.astype(np.uint8))
    normalized_img.save(f"{path2}/{splitext(t.split('/')[-1])[0]}.png")