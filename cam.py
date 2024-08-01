from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from torchvision.models import resnet50
import cv2
import glob
import numpy as np
import os
import torch
import torch
import torchvision
import torchvision.transforms.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import numpy as np
import sklearn.metrics as metrics
import torch.nn.functional as F
from config import *
import torch.nn.functional as F
from config import *
from PIL import Image
import random
from timesformer.models.vit import TimeSformer
from dataset import MyDataset_train, MyDataset_test, ExtraDataset
from FinalModel import FinalModel
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve,  auc, log_loss, confusion_matrix, accuracy_score, roc_auc_score
import copy
import os
import torch
import numpy as np
import cv2 as cv
import torchvision.transforms as transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
grad_block = []	
feaure_block = []	


def keyword(path):
    return int(path.split(os.path.sep)[-1].split('.')[0])

def backward_hook(module, grad_in, grad_out):
    
    print("backward_hook",grad_out[0].shape)
    grad_block.append(grad_out[0].detach())

def farward_hook(module, input, output):
    print("farward_hook",output.shape)
    feaure_block.append(output)

# 已知原图、梯度、特征图，开始计算可视化图
def cam_show_img(img, feature_map, grads,target_dir):
    print("in cam show",feature_map.shape,grads.shape) #64*768  128*128
    cam = np.zeros(feature_map.shape[0], dtype=np.float32)  # 二维，用于叠加
    grads = grads.reshape([grads.shape[0], -1]) 
    print(grads.shape) # 64*768
    # 梯度图中，每个通道计算均值得到一个值，作为对应特征图通道的权重
    weights = np.mean(grads, axis=1)	
    print(weights)#64
    print(feature_map[1,:].shape)
    for i, w in enumerate(weights):
        cam += w * np.sum(feature_map, axis=1)	# 特征图加权和
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam=cam.reshape((8,8))
    cam = cv2.resize(cam, (128, 128))
    # cam.dim=2 heatmap.dim=3
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)	# 伪彩色
    # img=img.transpose(0,2).numpy()
    cam_img = 0.4* heatmap + 0.6* img
    cv2.imwrite(target_dir, cam_img)


mod =TimeSformer(img_size=128, num_classes=2, num_frames=48,attention_type='divided_space_time').to(device)
unified_2D = transforms.Compose([
    transforms.Resize(us_shape),
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
]
)

state = torch.load("/data/data3/data_301/数据/20231019_mix/record_file/1019_Timesformer/72.pth")
mod.load_state_dict(state)
work_dir = "/data/data3/data_301/数据/20231019_mix/frame_12_36_HCC/20200101070002"
imgs_path = glob.glob(os.path.join(work_dir, "*"))
imgs_path.sort(key=keyword)
us_imgs = []

for j in imgs_path:
    us_img = cv2.imread(j)
    us_img = Image.fromarray(us_img.astype('uint8'))
    us_img = unified_2D(us_img)
    us_imgs.append(us_img)
rgb_img = torch.stack(us_imgs, dim=1)
rgb_img = torch.unsqueeze(rgb_img, 0).to(device)
mod.model.blocks[-1].norm1.register_forward_hook(farward_hook)
mod.model.blocks[-1].norm1.register_backward_hook(backward_hook)
# # forward 
# # 在前向推理时，会生成特征图和预测值
output = mod(rgb_img)
max_idx = np.argmax(output.cpu().data.numpy())
print("predict:{}".format(max_idx))

# # backward
mod.zero_grad()
class_loss = output[0, max_idx]	
class_loss.backward()	# 反向梯度，得到梯度图

# # grads
grads_val = grad_block[0].cpu().data.numpy()#.squeeze()
fmap = feaure_block[0].cpu().data.numpy()#.squeeze()

target="/data/data3/data_301/cam1020"
for i in range(len(imgs_path)):
    raw_img = cv2.imread(imgs_path[i])[:,:,::-1]
    raw_img=cv2.resize(raw_img, (128, 128))
    print(imgs_path[i])
    target_dir=os.path.join(target,imgs_path[i].split(os.path.sep)[-1])
    cam_show_img(raw_img, fmap[i,1:,:], grads_val[i,1:,:],target_dir)
