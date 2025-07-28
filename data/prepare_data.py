import os
from PIL import Image
import numpy as np

def resize_images(input_dir, output_dir, size=(32, 32)):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(input_dir, fname)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(size)
            img.save(os.path.join(output_dir, fname))

def main():
    raw_dir = "data/sample_images"
    processed_dir = "data/processed"
    resize_images(raw_dir, processed_dir)
    print(f"Processed images are saved in {processed_dir}")

if __name__ == '__main__':
    main()
