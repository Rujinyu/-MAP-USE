from timesformer.models.vit import TimeSformer
from  dataset import MyDataset_train,  MyDataset_test, ExtraDataset
import random
from config import *
import torch.nn.functional as F
import sklearn.metrics as metrics
import numpy as np
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms.transforms as transforms
import torchvision
import torch
import sys

sys.path.append("/data/data3/data_301/site-packages/")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False


def validity():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(60),
    ])
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    dataset_train = MyDataset_train(transform=transform_train)
    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, pin_memory=False if device == device else True, drop_last=False)

    dataset_test = MyDataset_test(transform=None)
    dataloader_test = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, pin_memory=False if device == device else True, drop_last=False)

    dataset_extra = ExtraDataset(transform=None)
    dataloader_extra = DataLoader(
        dataset_extra, batch_size=batch_size, shuffle=False, pin_memory=False if device == device else True, drop_last=False)

    model =TimeSformer(img_size=128, num_classes=2, num_frames=48,
                                  attention_type='divided_space_time',  pretrained_model='/data/data3/data_301/TimeSformer_divST_96x4_224_K600.pyth').to(device)

    state = torch.load("/data/data3/data_301/数据/20231019_mix/record_file/1019_Timesformer/136.pth")
    model.load_state_dict(state)
    learn_rate = 0.001
    
    loss = torch.nn.CrossEntropyLoss()
    all_score=[]
    all_label=[]
    all_part=[]
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, weight_decay=0.0001)  # weight_decay=0.00001)

    softmax_layer = torch.nn.Softmax(dim=-1)
    model.eval()
    y_true = []
    y_score = []
    y_part=[]
    l_sum = 0
    sum = 0
    for us_img, label, part in dataloader_train:
        # B * 2 * C * W * H
        us_img = us_img.to(device=device)
        label = label.to(device=device, dtype=torch.long).reshape(-1)
        with torch.no_grad():
            y = model(us_img)
        l = loss(y, label)
        sum = sum + 1
        l_sum = l_sum + l.item()
        y_score.append(softmax_layer(y.detach().cpu()).numpy())
        y_true.append(label.reshape(-1).detach().cpu().numpy())
        y_part.append(part)
    print(y_score)
    y_true = np.concatenate(y_true, 0)
    y_score = np.concatenate(y_score, 0)
    y_part=np.concatenate(y_part,0)
    all_score.extend(y_score[:,1])
    all_label.extend(y_true)
    all_part.extend(y_part)
    auc = metrics.roc_auc_score(y_true, y_score[:, 1])
    acc = np.sum(np.argmax(y_score, axis=1) == y_true) / y_true.shape[0]
    print("train, loss: {:.5f}, auc: {:.3f}, acc: {:.3f}".format( l_sum / sum, auc, acc))

    # test
    model.eval()
    y_true = []
    y_score = []
    y_part=[]
    l_sum = 0
    sum = 0
    for us_img, label, part in dataloader_test:
        # B * 2 * C * W * H
        us_img = us_img.to(device=device)
        label = label.to(device=device, dtype=torch.long).reshape(-1)
        with torch.no_grad():
            y = model(us_img)

        l = loss(y, label)
        sum = sum + 1
        l_sum = l_sum + l.item()
        y_score.append(softmax_layer(y.detach().cpu()).numpy())
        y_true.append(label.reshape(-1).detach().cpu().numpy())
        y_part.append(part)
    print(y_score)
    y_true = np.concatenate(y_true, 0)
    y_score = np.concatenate(y_score, 0)
    y_part=np.concatenate(y_part,0)
    all_score.extend(y_score[:,1])
    all_label.extend(y_true)
    all_part.extend(y_part)
    auc = metrics.roc_auc_score(y_true, y_score[:, 1])
    acc = np.sum(np.argmax(y_score, axis=1) == y_true) / y_true.shape[0]
    print("test , loss: {:.5f}, auc: {:.3f}, acc: {:.3f}".format(
         l_sum / sum, auc, acc))

    

    model.eval()
    y_true = []
    y_part=[]
    y_score = []
    l_sum = 0
    sum = 0
    for us_img, label, part in dataloader_extra:
        # B * 2 * C * W * H
        us_img = us_img.to(device=device)
        label = label.to(device=device, dtype=torch.long).reshape(-1)
        with torch.no_grad():
            y = model(us_img)
        l = loss(y, label)
        sum = sum + 1
        l_sum = l_sum + l.item()
        y_score.append(softmax_layer(y.detach().cpu()).numpy())
        y_true.append(label.reshape(-1).detach().cpu().numpy())
        y_part.append(part)
    print(y_score)
    y_true = np.concatenate(y_true, 0)
    y_score = np.concatenate(y_score, 0)
    y_part=np.concatenate(y_part,0)
    all_score.extend(y_score[:,1])
    all_label.extend(y_true)
    all_part.extend(y_part)
    auc = metrics.roc_auc_score(y_true, y_score[:, 1])
    acc = np.sum(np.argmax(y_score, axis=1) == y_true) / y_true.shape[0]
    print("extra , loss: {:.5f}, auc: {:.3f}, acc: {:.3f}".format( l_sum / sum, auc, acc))

    for i in range(len(all_label)):
        print(all_part[i],all_label[i],all_score[i])

set_seed(2023)
validity()
