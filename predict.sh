#!/bin/bash
#SBATCH --job-name=ntu60
#SBATCH --output=slurm_out15/%j.out
#SBATCH --error=slurm_out15/%j.err
#SBATCH --mem=10G                      
#SBATCH --time=00:40:00
#SBATCH --partition=l40s
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=vita

mkdir -p slurm_out15

source ~/anaconda3/bin/activate ada
nvidia-smi
python predict.py