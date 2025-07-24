# Zero-Shot Image Restoration with a Minimal Diffusion Model

- **Author:** Lin Xiaoya
- **Inspired by:** Prof. Wen Bihan’s CVPR 2025 work

## Project Overview

This project demonstrates zero-shot image denoising using a lightweight diffusion model, designed to run on CPU and entry-level GPUs.  
- Images: 32x32 or 64x64 pixels, 10–100 samples
- Model: Tiny UNet (fewer layers/filters)
- Tasks: Add noise, train on synthetic noise, test on unseen noise types (zero-shot)
- No dataset download required—uses a small sample set.

## How to Run

1. Clone this repo
2. Install dependencies:  
   `pip install -r requirements.txt`
3. Prepare data:  
   `python data/prepare_data.py`
4. Run notebooks step by step (Jupyter recommended)

## Folders

- `data/`: Small image set, data prep code
- `notebooks/`: Main experiment steps
- `src/`: Core code (model, diffusion steps, metrics)
- `results/`: Output images and plots

## Hardware

- Runs on most laptops (CPU or any GPU)
- Full experiment < 1 hour

## Acknowledgements

- [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) (reference)
