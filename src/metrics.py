import torch
import numpy as np

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def ssim(img1, img2):
    # For simplicity, use PSNR only in minimal version
    return None
