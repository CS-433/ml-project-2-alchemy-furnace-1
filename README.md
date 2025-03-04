# ML Project 2: Road Segmentation
For the second ML project of the course CS433 in EPFL, we are assigned to address the road segmentation challenge.
(full details: https://www.aicrowd.com/challenges/epfl-ml-road-segmentation)

The picture below shows one sample pair of an image and corresponding groundtruth.

![intoout](sample.png)

# Environment set up 

conda create --name project2 python=3.10

pip install -r requirements.txt

# Dataset
Training dataset: 100 pairs of real images and groundtruth images with the size 400*400 

Test dataset: 608*608

# Data Preprocessing and Augmentation

**Data cleaning**: remove the 11st, 28th, 92nd pairs of the images

**Data preprocessing**: python utils/preprocess_gt.py

**Data augment flip and rotate 90**: python utils/augment_data.py

**Data augment rotate 45**: first method in the report: python utils/augment_data_rotate2.py

**Data augment rotate 45**: second method in the report : python utils/augment_data_rotate.py

**Data split before training**: python utils/split_train_val.py


# Train
**python scripts/train.py --epochs 40 --batch-size 16 --learning-rate 3e-4  --scale 0.5 --validation 10.0  --classes 2**

or 

SCITAS: **sbatch wrapper.sh** (Change to train mode)

# Test (Predict)

You can download the pre-trained model ckpt.pth from (**https://drive.google.com/file/d/1MaREX4YBM7fRVr_5KS50bPBpuu6S4ZoQ/view?usp=drive_link**)

**python run.py** (you need Change the path of the checkpoints in **run.py**)

or 

SCITAS: **sbatch wrapper.sh** (Change to test mode)








