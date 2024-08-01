import copy
import os
import torch
import numpy as np
import cv2 as cv
import torchvision.transforms as transforms
from torch.utils.data import Dataset as pytorch_dataset
import glob
from PIL import Image
import pandas as pd
from config import *
import re
label_dir = "/data/data3/data_301/数据/20231019_mix/label_1019.xlsx"
label_content = pd.read_excel(label_dir)
def keyword(path): 
    return int(path.split(os.path.sep)[-1].split('.')[0])
class MyDataset_train(pytorch_dataset):
    def __init__(self, transform=None):
        super(MyDataset_train, self).__init__()
        work_dir = "/data/data3/data_301/数据/20231019_mix/frame_12_36"
        class1_path=[]
        self.datadir = work_dir
        label_0_path = glob.glob(os.path.join(self.datadir, "*"))
        for path in label_0_path:
            for item in range(len(label_content['trainID'])):
                if re.search(str(r'.*'+str(label_content['trainID'][item])), path.split(os.path.sep)[-1]):
                    class1_path.append(path)
           
        self.data_path = class1_path
        self.transform = transform
    def __len__(self):
        return len(self.data_path)
    def __getitem__(self, item):
        path_2D = self.data_path[item]
        ipath = path_2D
        # print(ipath)
        part = ipath.split(os.path.sep)[-1]
        unified_2D = transforms.Compose([
            transforms.Resize(us_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
        ]
        )
        us_imgs = []
        imgs_path = glob.glob(os.path.join(ipath, "*"))
        imgs_path.sort(key=keyword)
        for j in imgs_path:
            us_img = cv.imread(j)
            us_img = Image.fromarray(us_img.astype('uint8'))
            us_img = unified_2D(us_img)
            us_imgs.append(us_img)
        us_imgs = torch.stack(us_imgs, dim=1)
        label= None
        for item in range(len(label_content['trainID'])):
            if re.search(str(r'.*'+str(label_content['trainID'][item])), part):
                label = label_content['trainlabel'][item]
             
        return us_imgs, label, part

class MyDataset_test(pytorch_dataset):
    def __init__(self, transform=None):
        super(MyDataset_test, self).__init__()
        work_dir = "/data/data3/data_301/数据/20231019_mix/frame_12_36"
        class1_path=[]
        self.datadir = work_dir
        label_0_path = glob.glob(os.path.join(self.datadir, "*"))
        for path in label_0_path:
            for item in range(len(label_content['testID'])):
                if re.search(str(r'.*'+str(label_content['testID'][item])), path.split(os.path.sep)[-1]):
                    class1_path.append(path)
           
        self.data_path = class1_path
        self.transform = transform
    def __len__(self):
        return len(self.data_path)
    def __getitem__(self, item):
        path_2D = self.data_path[item]
        ipath = path_2D
        # print(ipath)
        part = ipath.split(os.path.sep)[-1]
        unified_2D = transforms.Compose([
            transforms.Resize(us_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
        ]
        )
        us_imgs = []
        imgs_path = glob.glob(os.path.join(ipath, "*"))
        imgs_path.sort(key=keyword)
        for j in imgs_path:
            us_img = cv.imread(j)
            us_img = Image.fromarray(us_img.astype('uint8'))
            us_img = unified_2D(us_img)
            us_imgs.append(us_img)
        us_imgs = torch.stack(us_imgs, dim=1)
        label= None
        for item in range(len(label_content['testID'])):
            if re.search(str(r'.*'+str(label_content['testID'][item])), part):
                label = label_content['testlabel'][item]
             
        return us_imgs, label, part

extra_excel = "/data/data3/data_301/Label（外部验证补充变量）.xlsx"
extra_content = pd.read_excel(extra_excel)

class ExtraDataset(pytorch_dataset):

    def __init__(self, transform=None):
        super(ExtraDataset, self).__init__()
        work_dir = "/data/data3/data_301/数据/20231019_mix/frame_12_36_extra"
        class1_path=[]
        self.datadir = work_dir
        label_0_path = glob.glob(os.path.join(self.datadir, "*"))
        for path in label_0_path:
            for item in range(len(label_content['extraID'])):
                if re.search(str(r'.*'+str(label_content['extraID'][item])), path.split(os.path.sep)[-1]):
                    class1_path.append(path)
           
        self.data_path = class1_path
        self.transform = transform
    def __len__(self):
        return len(self.data_path)
    def __getitem__(self, item):
        path_2D = self.data_path[item]
        ipath = path_2D
        # print(ipath)
        part = ipath.split(os.path.sep)[-1]
        unified_2D = transforms.Compose([
            transforms.Resize(us_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
        ]
        )
        us_imgs = []
        imgs_path = glob.glob(os.path.join(ipath, "*"))
        imgs_path.sort(key=keyword)
        for j in imgs_path:
            us_img = cv.imread(j)
            us_img = Image.fromarray(us_img.astype('uint8'))
            us_img = unified_2D(us_img)
            us_imgs.append(us_img)
        us_imgs = torch.stack(us_imgs, dim=1)
        label= None
        for item in range(len(label_content['extraID'])):
            if re.search(str(r'.*'+str(label_content['extraID'][item])), part):
                label = label_content['extralabel'][item]
             
        return us_imgs, label, part
class ExtraDataset2(pytorch_dataset):
    def __init__(self, transform=None):
        super(ExtraDataset2, self).__init__()
        work_dir = "/data/data3/data_301/数据/20231019_mix/frame_12_36_HCC"
        class1_path=[]
        self.datadir = work_dir
        label_0_path = glob.glob(os.path.join(self.datadir, "*"))
        for path in label_0_path:
            for item in range(len(label_content['extraheID'])):
                if re.search(str(r'.*'+str(label_content['extraheID'][item])), path.split(os.path.sep)[-1]):
                    class1_path.append(path)
           
        self.data_path = class1_path
        self.transform = transform
    def __len__(self):
        return len(self.data_path)
    def __getitem__(self, item):
        path_2D = self.data_path[item]
        ipath = path_2D
        part = ipath.split(os.path.sep)[-1]
        unified_2D = transforms.Compose([
            transforms.Resize(us_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
        ]
        )
        us_imgs = []
        imgs_path = glob.glob(os.path.join(ipath, "*"))
        imgs_path.sort(key=keyword)
        for j in imgs_path:
            us_img = cv.imread(j)
            us_img = Image.fromarray(us_img.astype('uint8'))
            us_img = unified_2D(us_img)
            us_imgs.append(us_img)
        us_imgs = torch.stack(us_imgs, dim=1)
        label= None
        for item in range(len(label_content['extraheID'])):
            if re.search(str(r'.*'+str(label_content['extraheID'][item])), part):
                label = label_content['extrahelabel'][item]
             
        return us_imgs, label, part


        