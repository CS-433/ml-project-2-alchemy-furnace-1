import numpy as np
from PIL import Image
import torch
from pathlib import Path
from os.path import splitext

file = '/home/yeguo/MLp2/outputs/checkpoint_epoch20_0.7186881899833679/test_1_OUT.png'
img = Image.open(file)
print(img.size)