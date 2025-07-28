import torch
import torch.nn as nn

class TinyUNet(nn.Module):
    def __init__(self, in_channels=3, base=32):
        super().__init__()
        self.enc1 = nn.Conv2d(in_channels, base, 3, padding=1)
        self.enc2 = nn.Conv2d(base, base*2, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dec1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.outc = nn.Conv2d(base, in_channels, 1)
        self.act = nn.ReLU()

    def forward(self, x, t=None):
        x1 = self.act(self.enc1(x))
        x2 = self.act(self.enc2(self.pool(x1)))
        x3 = self.act(self.dec1(x2))
        x4 = self.outc(x3 + x1)
        return x4
