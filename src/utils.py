import torch
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image

def save_sample(img_tensor, path):
    img_tensor = img_tensor.clamp(0, 1)
    save_image(img_tensor, path)

def plot_side_by_side(orig, noisy, denoised, save_path):
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    for ax, img, title in zip(axs, [orig, noisy, denoised], ['Original', 'Noisy', 'Denoised']):
        ax.imshow(img.permute(1, 2, 0).cpu().numpy())
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
