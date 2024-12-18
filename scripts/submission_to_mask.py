#!/usr/bin/python
import os
import sys
import math
import matplotlib.image as mpimg
import numpy as np
import cv2  # 使用OpenCV替代Image库

label_file = 'datasets/sample_submission.csv'

h = 16
w = h
imgwidth = int(math.ceil((600.0/w))*w)
imgheight = int(math.ceil((600.0/h))*h)
nc = 3

# Convert an array of binary labels to a uint8
def binary_to_uint8(img):
    rimg = (img * 255).round().astype(np.uint8)
    return rimg

def reconstruct_from_labels(image_id):
    im = np.zeros((imgwidth, imgheight), dtype=np.uint8)
    f = open(label_file)
    lines = f.readlines()
    # 将image_id格式化为三位数，并在其后加上下划线
    # 例如，如果image_id是5，则image_id_str将是'005_'
    image_id_str = '%.3d_' % image_id
    for i in range(1, len(lines)):
        line = lines[i]
        if not image_id_str in line:
            continue

        tokens = line.split(',')
        id = tokens[0]
        prediction = int(tokens[1])
        tokens = id.split('_')
        i = int(tokens[1])
        j = int(tokens[2])

        je = min(j+w, imgwidth)
        ie = min(i+h, imgheight)
        if prediction == 0:
            adata = np.zeros((w,h))
        else:
            adata = np.ones((w,h))

        im[j:je, i:ie] = binary_to_uint8(adata)

    # 使用OpenCV保存图像
    cv2.imwrite('prediction_' + '%.3d' % image_id + '.png', im)

    return im

for i in range(1,2):
    reconstruct_from_labels(i)
