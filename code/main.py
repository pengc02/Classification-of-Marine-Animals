"""
# File       : main.py
# Time       ：2022/11/23 12:47
# Author     ：Peng Cheng
# Description：The main function to control config and call the training or testing process
"""

import argparse
import os
import yaml
from pprint import pprint
from easydict import EasyDict
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
import torch
import cv2
from trainer import train,test
from model.MyNet import *
from model.MyNetPro import *
from model.networks import *
from dataset import *
import random
import wandb
from torchvision import models

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch implementation of Classification')
    parser.add_argument('--config', default='',help='config file path')
    # exclusive arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true',help='train mode')
    group.add_argument('--test', action='store_true',help='test mode')

    parser.add_argument('--model', default='MyNetPro', help='Model Name')

    return parser.parse_args()


def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(config,classifier):

    if config.train:
        train(config,classifier)
    elif config.test:
        model_path = config.model_dir + '/' + config.model + '.pt'
        classifier.load_state_dict(torch.load(model_path))
        print(model_path)
        test(config,classifier)

        # model_path = config.save_dir 
        # model_list = os.listdir(model_path)
        # model_list.sort(key=lambda x: int(x[5:-3]))  ##文件名按数字排序
        # classifier.load_state_dict(torch.load(model_path + '/' + model_list[-1]))
        # print(model_path + '/' + model_list[-1])
        # test(config, classifier)


if __name__ == '__main__':
    wandb.init(project="project2-report", entity="pengc02",name='mynetpro')

    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    for k, v in vars(args).items():
        config[k] = v
    config = EasyDict(config)
    
    if config.model == 'MyNet':
        classifier = MyNet()
    elif config.model == 'MyNetPro':
        classifier = MyNetPro()
    elif config.model == 'VGG16':
        classifier = vgg16_bn()
    elif config.model == 'VGG16-pretrain':
        classifier = vgg16_bn_pre()
    elif config.model == 'ResNet34':
        classifier = resnet34()
    elif config.model == 'ResNet34-pretrain':
        classifier = resnet34pre()
    elif config.model == 'ResNet34-pretrain-frozen':
        classifier = resnet34pre_frozen()
    else:
        print("wrong model name,change to mynetpro")
        classifier = MyNetPro()

    main(config,classifier)
