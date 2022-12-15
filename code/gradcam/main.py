# -*- coding: UTF-8 -*-
"""
@File ：gradcam.py
@Author ：PengCheng
@Date ：2022/12/11 21:30
@Intro:
"""
import os
import PIL
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image

from utils import visualize_cam, Normalize
from gradcam import GradCAM, GradCAMpp
from torchvision import transforms

img_dir = 'images'
img_name = '4.jpg'
img_path = os.path.join(img_dir, img_name)

pil_img = PIL.Image.open(img_path)


normalizer = Normalize(mean=[0.437, 0.42, 0.344], std=[0.186, 0.185, 0.187])
torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
torch_img = F.upsample(torch_img, size=(224, 224), mode='bilinear', align_corners=False)
normed_torch_img = normalizer(torch_img)

alexnet = models.alexnet(pretrained=True)
alexnet.eval()
alexnet.cuda()

resnet = models.resnet34(pretrained=True)
numFit = resnet.fc.in_features
resnet.fc = nn.Linear(numFit, 19)
resnet.load_state_dict(torch.load('./modelzoo/resnet34-pre-frozen.pt'))
resnet.eval()
resnet.cuda()
print(resnet)
cam_dict = dict()


resnet_model_dict = dict(type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
resnet_gradcam = GradCAM(resnet_model_dict, True)
resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
cam_dict['resnet'] = [resnet_gradcam, resnet_gradcampp]

# feed forward image
images = []
for gradcam, gradcam_pp in cam_dict.values():
    mask, _ = gradcam(normed_torch_img)
    heatmap, result = visualize_cam(mask.cpu(), torch_img.cpu())

    mask_pp, _ = gradcam_pp(normed_torch_img)
    heatmap_pp, result_pp = visualize_cam(mask_pp.cpu(), torch_img.cpu())

    images.append(torch.stack([torch_img.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))

images = make_grid(torch.cat(images, 0), nrow=5)

output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)
output_name = img_name
output_path = os.path.join(output_dir, output_name)

save_image(images, output_path)
PIL.Image.open(output_path)