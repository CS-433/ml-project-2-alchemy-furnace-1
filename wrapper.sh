#!/bin/bash
#SBATCH --job-name=ntu60
#SBATCH --output=slurm_out1/%j.out
#SBATCH --error=slurm_out1/%j.err
#SBATCH --mem=10G                      
#SBATCH --time=00:40:00
#SBATCH --partition=h100
#SBATCH --cpus-per-task=40
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --account=vita

mkdir -p slurm_out1

source ~/anaconda3/bin/activate project2
nvidia-smi

# train mode
# python scripts/train.py --epochs 40 --batch-size 16 --learning-rate 3e-4  --scale 0.5 --validation 10.0  --classes 2 #--amp --bilinear

# test mode
python predict.py