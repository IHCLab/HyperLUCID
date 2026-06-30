"""
HyperLUCID Utility Functions

Code implemented by:
    Shih-Min Hsu
    Ching-Yun Liang
"""
import torch
import torch.nn as nn
from torchvision.models import *


class Calibration_Function(nn.Module):
    def __init__(self, channel: int, model='normal'):
        super(Calibration_Function, self).__init__()
        self.model=model 
        self.channel = channel
        self.group_channel = 3 * (channel // 4)
        self.res_channel = channel - self.group_channel
        self.Elu = nn.ELU()
        self.Encorder_1 = nn.Sequential(
            nn.Conv2d(self.group_channel, 48, kernel_size=1, padding=0, groups=3),
            self.Elu
            )
        self.Encorder_2 = nn.Sequential(
            nn.Conv2d(self.res_channel, 16, kernel_size=1, padding=0),
            self.Elu
            )
        self.Decorder = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=1, padding=0),
            self.Elu
            )

    def forward(self, x: torch.Tensor):
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)
        x_1, x_2 = torch.split(x, [self.group_channel, self.res_channel], dim=1)
        x_1 = self.Encorder_1(x_1)
        x_2 = self.Encorder_2(x_2)
        x_res = torch.cat([x_1,x_2], 1)
        
        x_res = self.Decorder(x_res)
        x = x_res + x
        x = x.squeeze(0)
        x = x.permute(1, 2, 0)
        return x
