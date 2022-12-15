import torch
import torch.nn as nn
import torchvision

def vgg16_bn():
    vgg16 = torchvision.models.vgg16_bn()
    vgg16.classifier.add_module("head", nn.Linear(1000, 19))
    return vgg16

def vgg16_bn_pre():
    vgg16 = torchvision.models.vgg16_bn(pretrained=True)
    vgg16.classifier.add_module("head", nn.Linear(1000, 19))
    return vgg16

def resnet34():
    # 用个预训练
    res34 = torchvision.models.resnet34(pretrained=False)
    numFit = res34.fc.in_features
    res34.fc = nn.Linear(numFit, 19)
    print("resnet 34 down")
    return res34

def resnet34pre():
    # 用个预训练
    res34 = torchvision.models.resnet34(pretrained=True)
    numFit = res34.fc.in_features
    res34.fc = nn.Linear(numFit, 19)
    print("resnet 34 down")
    return res34

def resnet34pre_frozen():
    # 用个预训练
    res34 = torchvision.models.resnet34(pretrained=True)
    numFit = res34.fc.in_features
    res34.fc = nn.Linear(numFit, 19)
    for param in res34.parameters():
        param.requires_grad = False
    for param in res34.fc.parameters():
        param.requires_grad = True
    for param in res34.layer4.parameters():
        param.requires_grad = True
    print("resnet 34 down")
    return res34

def resnext101pre_frozen():
    resnext101 =torchvision.models.resnext101_32x8d(pretrained=True)
    resnext101.add_module("head", nn.Linear(1000, 19))
    # print(resnext101)
    for param in resnext101.parameters():
        param.requires_grad = False
    for param in resnext101.fc.parameters():
        param.requires_grad = True
    for param in resnext101.head.parameters():
        param.requires_grad = True
    return resnext101

def bitpre():
    import timm
    bit = timm.create_model('resnetv2_101x1_bitm', pretrained=True, num_classes=19)
    return bit



