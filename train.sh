#!/bin/bash
#SBATCH --job-name=ntu60
#SBATCH --output=slurm_out11/%j.out
#SBATCH --error=slurm_out11/%j.err
#SBATCH --mem=80G                      
#SBATCH --time=00:20:00
#SBATCH --partition=h100
#SBATCH --cpus-per-task=40
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=vita

mkdir -p slurm_out11

source ~/anaconda3/bin/activate ada
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export OMP_NUM_THREADS=1
echo "Running on host: $HOSTNAME"

srun python -u train.py
