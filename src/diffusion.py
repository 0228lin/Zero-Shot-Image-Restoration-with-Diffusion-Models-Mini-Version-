import torch
import torch.nn.functional as F
import numpy as np

class SimpleDiffusion:
    def __init__(self, timesteps=100, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.timesteps = timesteps
        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, 0)

    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alpha_bar = self.alpha_bar[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus = (1 - self.alpha_bar[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise, noise

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.timesteps, (batch_size,))
