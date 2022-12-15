import numpy as np
import torch
import cv2
import os
from PIL import Image

loadpath = '/DATA/disk1/pengcheng/homework/project2/data'
outputpath = '/DATA/disk1/pengcheng/homework/project2/database'

def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


def load_data(loadpath,savepath):
    path = loadpath
    dirs = os.listdir(loadpath)
    for f in dirs:
        if os.path.isfile(os.path.join(path, f)):
            dirs.remove(f)

    train_data = []
    valid_data = []
    test_data = []

    ratio1 = 0.9
    ratio2 = 8/9

    for i,dir in enumerate(dirs):
        print(dir)
        if dir == 'Turtle_Tortoise' or dir == 'Whale':
            dpath = path + '/' + dir
            files = os.listdir(dpath)
            data = []
            for i, f in enumerate(files):
                if dir == 'Turtle_Tortoise':
                    if i == 500:
                        break
                fpath = dpath + '/' + f
                img = Image.open(fpath)
                data.append(img)
                # with open(fpath, 'rb') as f:
                #     img = Image.open(f)
                #     data.append(img)

            train_and_valid_data, test_data_tmp = data_split(data, ratio1)
            train_data_tmp, valid_data_tmp = data_split(train_and_valid_data, ratio2)

            trainpath = savepath + '/train/' + dir
            validpath = savepath + '/valid/' + dir
            testpath = savepath + '/test/' + dir

            if not os.path.exists(trainpath):
                os.makedirs(trainpath)
            if not os.path.exists(validpath):
                os.makedirs(validpath)
            if not os.path.exists(testpath):
                os.makedirs(testpath)

            for i, img in enumerate(train_data_tmp):
                img.save(trainpath + '/' + dir + f'_{i}.png')
            for i, img in enumerate(valid_data_tmp):
                img.save(validpath + '/' + dir + f'_{i}.png')
            for i, img in enumerate(test_data_tmp):
                img.save(testpath + '/' + dir + f'_{i}.png')

            train_data += train_data_tmp
            valid_data += valid_data_tmp
            test_data += test_data_tmp
            # 对于同一个文件夹的，应该进行分割



    return train_data,valid_data,test_data


load_data(loadpath,outputpath)