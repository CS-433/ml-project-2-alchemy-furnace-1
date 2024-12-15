#!/bin/bash
#SBATCH --job-name=ntu60
#SBATCH --output=slurm_out11/%j.out
#SBATCH --error=slurm_out11/%j.err
#SBATCH --mem=40G                      
#SBATCH --time=01:40:00
#SBATCH --partition=h100
#SBATCH --cpus-per-task=40
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --account=vita

mkdir -p slurm_out11

source ~/anaconda3/bin/activate ada
nvidia-smi

# 执行 Python 脚本
python scripts/train.py --epochs 100 --batch-size 40 --learning-rate 5e-4  --scale 0.5 --validation 5.0  --classes 2 #--amp --bilinear
# python predict.py