import os
import shutil

# 获取当前目录下所有文件
files = os.listdir('.')

# 创建目标目录
target_dir = './slurm_logs'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 遍历文件列表，将所有类似slurm-2483233.out的文件移动到目标目录
for file in files:
    if file.startswith('slurm-') and file.endswith('.out'):
        shutil.move(file, target_dir)
