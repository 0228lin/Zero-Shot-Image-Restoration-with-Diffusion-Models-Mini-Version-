import kagglehub
import os
import shutil
import random

def download_cifar100():
    print("Downloading CIFAR100 dataset from KaggleHub...")
    path = kagglehub.dataset_download("melikechan/cifar100")
    print("Downloaded to:", path)
    return path

def sample_images(cifar_path, dst_dir="data/sample_images", n_classes=10, images_per_class=5, seed=42):
    random.seed(seed)
    train_dir = os.path.join(cifar_path, "train")
    os.makedirs(dst_dir, exist_ok=True)

    # List class folders
    classes = [c for c in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, c))]
    random.shuffle(classes)
    selected_classes = classes[:n_classes]

    print(f"Sampling {images_per_class} images from each of {n_classes} classes...")
    total = 0
    for cls in selected_classes:
        cls_dir = os.path.join(train_dir, cls)
        imgs = [f for f in os.listdir(cls_dir) if f.endswith('.png')]
        chosen_imgs = random.sample(imgs, min(images_per_class, len(imgs)))
        for img in chosen_imgs:
            src_img = os.path.join(cls_dir, img)
            dst_img = os.path.join(dst_dir, f"{cls}_{img}")
            shutil.copy(src_img, dst_img)
            total += 1
    print(f"Sampled {total} images to '{dst_dir}'.")

if __name__ == '__main__':
    cifar_path = download_cifar100()
    sample_images(cifar_path)
