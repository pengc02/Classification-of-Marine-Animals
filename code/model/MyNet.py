"""
# File       : model.py
# Time       ：2022/11/23 16:06
# Author     ：Peng Cheng
# Description：
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3,10, kernel_size=(5, 5)),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, kernel_size=(5, 5)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(56180, 500),
            nn.ReLU(),
            nn.Linear(500, 19),
        )

    def forward(self, x):
        x = self.layers(x)
        return x