"""
# File       : dataset.py
# Time       ：2022/11/23 16:51
# Author     ：Peng Cheng
# Description：
"""

import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import json
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder


def get_SeaAnimal_dataloader(config,type):
    if type == 'train':
        transform_pro = transforms.Compose([
            transforms.Resize((300, 300)),  # 缩放
            transforms.RandomResizedCrop(250, scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333),
                                                     interpolation=2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(45),
            transforms.RandomGrayscale(p=0.5), #随机灰度化
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # 转为张量，同时归一化，即除255
            transforms.Normalize(mean=[0.431, 0.402, 0.293], std=[0.181, 0.184, 0.179]),  # 标准化
        ])
        transform = transforms.Compose([
            transforms.Resize((config.data.resize, config.data.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.431, 0.402, 0.293], std=[0.181, 0.184, 0.179]),
        ])
        train_dataset = ImageFolder('database/train_pro/',transform_pro)

        train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle = True, num_workers=4,drop_last=False)
        return train_dataloader
    elif type == 'valid':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.417, 0.395, 0.316], [0.182, 0.184, 0.181]),
        ])
        valid_dataset = ImageFolder('database/valid/', transform)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config.train_batch_size, shuffle = True, num_workers=4,drop_last=False)
        return valid_dataloader
    else:

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.437, 0.42, 0.344], [0.186, 0.185, 0.187]),
        ])
        test_dataset = ImageFolder('database/test/', transform)
        # print(test_dataset.class_to_idx)  # 打印标签和类别的对应关系
        test_dataloader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=True, num_workers=0)
        return test_dataloader

