from timesformer.models.vit import TimeSformer
from  dataset import MyDataset_train,  MyDataset_test, ExtraDataset,ExtraDataset2
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


def train_test():
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
   
    dataset_extra2 = ExtraDataset2(transform=None)
    dataloader_extra2 = DataLoader(
        dataset_extra2, batch_size=batch_size, shuffle=False, pin_memory=False if device == device else True, drop_last=False)

    model =TimeSformer(img_size=128, num_classes=2, num_frames=48,
                                  attention_type='divided_space_time',  pretrained_model='/data/data3/data_301/TimeSformer_divST_96x4_224_K600.pyth').to(device)
   
    filename = "/data/data3/data_301/数据/20231019_mix/record_file/1112_Timesformer/record_models_timesformer_dep1_head4_drop1_1111.txt"  # depth1batch8embed768
    learn_rate = 0.001
  
    loss = torch.nn.CrossEntropyLoss()
    pre_auc = 0
    pre_acc = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, weight_decay=0.0001)  # weight_decay=0.00001)

    for epoch in range(epoches):

        model.train()
        for us_img, label,part in dataloader_train:
            # B * 2 * C * W * H
            us_img = us_img.to(device=device)
            print("us_img shape",us_img.shape)
            label = label.to(device=device, dtype=torch.long)
            y = model(us_img)
            l = loss(y, label)
            print(l)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        softmax_layer = torch.nn.Softmax(dim=-1)

        model.eval()
        y_true = []
        y_score = []
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
        print(y_score)
        y_true = np.concatenate(y_true, 0)
        y_score = np.concatenate(y_score, 0)

        auc = metrics.roc_auc_score(y_true, y_score[:, 1])

        acc = np.sum(np.argmax(y_score, axis=1) == y_true) / y_true.shape[0]
        print("epoch: {} train, loss: {:.5f}, auc: {:.3f}, acc: {:.3f}".format(
            epoch, l_sum / sum, auc, acc))
        with open(filename, 'a') as fp:
            fp.write("epoch: {} train, loss: {:.5f}, auc: {:.3f}, acc: {:.3f}".format(epoch, l_sum / sum, auc, acc))
            fp.write("\n")

        # test
        model.eval()
        y_true = []
        y_score = []
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
        y_true = np.concatenate(y_true, 0)
        y_score = np.concatenate(y_score, 0)
        auc = metrics.roc_auc_score(y_true, y_score[:, 1])

        acc = np.sum(np.argmax(y_score, axis=1) == y_true) / y_true.shape[0]
        print("epoch: {} test , loss: {:.5f}, auc: {:.3f}, acc: {:.3f}".format(
            epoch, l_sum / sum, auc, acc))
        with open(filename, 'a') as fp:
            fp.write("epoch: {} test, loss: {:.5f}, auc: {:.3f}, acc: {:.3f}".format(
                epoch, l_sum / sum, auc, acc))
            fp.write("\n")

        

        model.eval()
        y_true = []
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

        y_true = np.concatenate(y_true, 0)
        y_score = np.concatenate(y_score, 0)
        auc = metrics.roc_auc_score(y_true, y_score[:, 1])

        acc = np.sum(np.argmax(y_score, axis=1) == y_true) / y_true.shape[0]
        print("epoch: {} extra , loss: {:.5f}, auc: {:.3f}, acc: {:.3f}".format(
            epoch, l_sum / sum, auc, acc))
        with open(filename, 'a') as fp:
            fp.write("epoch: {} extra, loss: {:.5f}, auc: {:.3f}, acc: {:.3f}".format(
                epoch, l_sum / sum, auc, acc))
            fp.write("\n")

        
        model.eval()
        y_true = []
        y_score = []
        l_sum = 0
        sum = 0
        for us_img, label, part in dataloader_extra2:
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

        y_true = np.concatenate(y_true, 0)
        y_score = np.concatenate(y_score, 0)
        auc = metrics.roc_auc_score(y_true, y_score[:, 1])

        acc = np.sum(np.argmax(y_score, axis=1) == y_true) / y_true.shape[0]
        print("epoch: {} extra2 , loss: {:.5f}, auc: {:.3f}, acc: {:.3f}".format(
            epoch, l_sum / sum, auc, acc))
        with open(filename, 'a') as fp:
            fp.write("epoch: {} extra2, loss: {:.5f}, auc: {:.3f}, acc: {:.3f}".format(
                epoch, l_sum / sum, auc, acc))
            fp.write("\n")
        state_dict = model.state_dict()
        torch.save(state_dict, "/data/data3/data_301/数据/20231019_mix/record_file/1112_Timesformer/{}.pth".format(epoch))
       


set_seed(2023)
train_test()
