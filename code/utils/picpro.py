import torch
from PIL import Image
import os
import random

def image_pro(path):
    i = path
    image0 = Image.open(i)

    image1 = Image.open(i)
    image1 = image1.rotate(45)
    image1 = image1.resize((256, 256))

    image2 = Image.open(i)
    image2 = image2.transpose(Image.FLIP_LEFT_RIGHT)
    image2 = image2.resize((256, 256))
    # 随机裁剪
    image3 = Image.open(i)
    image3 = image3.resize((384, 384))
    x_min = random.randint(0, 128)
    y_min = random.randint(0, 128)
    box = (x_min, y_min, x_min + 256, y_min + 256)
    image3 = image3.crop(box)
    return image0, image1,image2,image3


# 对于每一张图片进行三种增强：
# 1. 旋转+resize
# 2. 镜像+ resize
# 3. resize

def load_pic():
    path = '/DATA/disk1/pengcheng/homework/project2/database/train'
    newpath = '/DATA/disk1/pengcheng/homework/project2/database/train_pro'
    dirs = os.listdir(path)
    for f in dirs:
        if os.path.isfile(os.path.join(path, f)):
            dirs.remove(f)

    for i, dir in enumerate(dirs):
        print(dir)
        dpath = path + '/' + dir
        files = os.listdir(dpath)
        print(i)
        print(dpath)
        for f in files:
            fpath = dpath + '/' + f
            fn = f[:-4]
            if 'pro' in fn:
                pass
            else:
                image0 = Image.open(fpath)
                propath = newpath + '/' + dir
                if not os.path.exists(propath):
                    os.makedirs(propath)
                image0.save(propath + '/' + f'{fn}.png')
            # img0,img1,img2,img3 = image_pro(fpath)

            # propath = newpath + '/' + dir
            # if not os.path.exists(propath):
            #     os.makedirs(propath)
            # img0.save(dpath + '/' + f'{fn}.png')
            # img1.save(dpath + '/' + f'{fn}_pro1.png')
            # img2.save(dpath + '/' + f'{fn}_pro2.png')
            # img3.save(dpath + '/' + f'{fn}_pro3.png')

            # image0 = Image.open(fpath)
            # propath = newpath + '/' + dir
            # if not os.path.exists(propath):
            #     os.makedirs(propath)
            # image0.save(propath + '/' + f'{fn}.png')



load_pic()