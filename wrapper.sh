#!/bin/bash

#SBATCH --partition=debug         # 使用 debug 分区
#SBATCH --qos=debug               # 指定 QOS 为 debug
#SBATCH --nodes=1                 # 使用 1 个节点（默认是 i63）
#SBATCH --ntasks=1                # 单任务
#SBATCH --cpus-per-task=4         # 每个任务使用 4 个 CPU
#SBATCH --gres=gpu:2
#SBATCH --mem=16G                 # 请求 16GB 内存
#SBATCH --time=00:59:59           # 作业最大运行时间

export CUDA_VISIBLE_DEVICES=0,1
nvidia-smi -l 10 > gpu_usage.log &

source ~/miniconda3/bin/activate base

#python scripts/train.py --epochs 30 --batch-size 2 --learning-rate 2e-4 --weight-decay 1e-5  --scale 0.5 --validation 10.0  --classes 2 #--amp --bilinear
python predict.py

