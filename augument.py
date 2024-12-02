import os
from PIL import Image

def create_dirs(base_dirs, operations):
    for base in base_dirs:
        for op in operations:
            os.makedirs(os.path.join(base, op), exist_ok=True)

def augment_images(input_dir, output_dirs, operations):
    for filename in os.listdir(input_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            filepath = os.path.join(input_dir, filename)
            image = Image.open(filepath)
            
            for op in operations:
                if op == 'flipud':
                    augmented_image = image.transpose(Image.FLIP_TOP_BOTTOM)
                elif op == 'fliplr':
                    augmented_image = image.transpose(Image.FLIP_LEFT_RIGHT)
                elif op == 'rotate_90':
                    augmented_image = image.rotate(90, expand=True)
                elif op == 'rotate_180':
                    augmented_image = image.rotate(180, expand=True)
                elif op == 'rotate_270':
                    augmented_image = image.rotate(270, expand=True)
                else:
                    continue
                
                output_path = os.path.join(output_dirs[op], filename)
                augmented_image.save(output_path)
# define the input directories and operations
base_input_dirs = ['training/groundtruth', 'training/images']
operations = ['flipud', 'fliplr', 'rotate_90', 'rotate_180', 'rotate_270']

# create the output directories
output_dirs = {}
for base in base_input_dirs:import os
from PIL import Image

def create_dirs(base_dirs, operations):
    for base in base_dirs:
        for op in operations:
            dir_name = f"{base}_{op}"
            os.makedirs(dir_name, exist_ok=True)

def augment_images(input_dir, operations):
    for filename in os.listdir(input_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            filepath = os.path.join(input_dir, filename)
            image = Image.open(filepath)
            
            for op in operations:
                if op == 'flip1':
                    augmented_image = image.transpose(Image.FLIP_TOP_BOTTOM)
                elif op == 'flip2':
                    augmented_image = image.transpose(Image.FLIP_LEFT_RIGHT)
                elif op == 'rotate_90':
                    augmented_image = image.rotate(90, expand=True)
                elif op == 'rotate_180':
                    augmented_image = image.rotate(180, expand=True)
                elif op == 'rotate_270':
                    augmented_image = image.rotate(270, expand=True)
                else:
                    continue
                
                output_dir = f"{input_dir}_{op}"
                output_path = os.path.join(output_dir, filename)
                augmented_image.save(output_path)

# define the input directories and operations
base_input_dirs = ['./training/groundtruth', './training/images']
operations = ['flip1', 'flip2', 'rotate_90', 'rotate_180', 'rotate_270']

# create the output directories
create_dirs(base_input_dirs, operations)


for base in base_input_dirs:
    augment_images(base, operations)

print("Data augmentation completed.")


