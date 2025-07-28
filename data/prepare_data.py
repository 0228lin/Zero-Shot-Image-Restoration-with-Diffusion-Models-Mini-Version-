import os
from PIL import Image
import numpy as np

def resize_and_save(img_path, out_dir, size=(32, 32)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(size)
    img.save(os.path.join(out_dir, os.path.basename(img_path)))

def main():
    os.makedirs('data/sample_images', exist_ok=True)
    # Place 10-100 random images (e.g., personal photos, standard test images) into this folder.
    # If you want to automate, download from a URL or use torchvision.datasets.CIFAR10.
    print("Place your sample images in data/sample_images/ and rerun this script if needed.")

if __name__ == '__main__':
    main()
