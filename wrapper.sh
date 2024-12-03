#!/bin/bash
#SBATCH --partition=debug         # 使用 debug 分区
#SBATCH --qos=debug               # 指定 QOS 为 debug
#SBATCH --nodes=1                 # 使用 1 个节点（默认是 i63）
#SBATCH --ntasks=1                # 单任务
#SBATCH --cpus-per-task=4         # 每个任务使用 4 个 CPU
#SBATCH --gres=gpu:2
#SBATCH --mem=16G                 # 请求 16GB 内存
#SBATCH --time=01:00:00           # 作业最大运行时间
nvidia-smi
source ~/miniconda3/bin/activate base
# 执行 Python 脚本
#python scripts/train.py --epochs 20 --batch-size 1 --learning-rate 2e-5  --scale 0.5 --validation 10.0  --classes 2 #--amp --bilinear
python predict.py