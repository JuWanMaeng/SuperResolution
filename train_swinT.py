import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import torch.nn as nn
from models import swin_transformer,ViT
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import copy
import time


# get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

# calculate the metric per mini-batch
def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects


# calculate the loss per mini-batch
def loss_batch(loss_func, output, target, opt=None):
    loss_b = loss_func(output, target)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()
    
    return loss_b.item(), metric_b


# calculate the loss per epochs
def loss_epoch(model, loss_func, dataset_dl, phase=None, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)
    if phase=='train':
        tq=tqdm(dataset_dl, ncols=80, smoothing=0, bar_format='train: {desc}|{bar}{r_bar}')
    else:
        tq=tqdm(dataset_dl, ncols=80, smoothing=0, bar_format='train: {desc}|{bar}{r_bar}')
    for xb, yb in tq:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)

        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

        running_loss += loss_b
        
        if metric_b is not None:
            running_metric += metric_b

  
    loss = running_loss / len_data
    metric = running_metric / len_data
    return loss, metric



path2data = 'dataset/STL'

# load dataset
train_ds = datasets.STL10(path2data, split='train', download=False, transform=transforms.ToTensor())
val_ds = datasets.STL10(path2data, split='test', download=False, transform=transforms.ToTensor())

# define transformation
transformation = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(224),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# apply transformation to dataset
train_ds.transform = transformation
val_ds.transform = transformation

# make dataloade
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True,num_workers=8)
val_dl = DataLoader(val_ds, batch_size=8, shuffle=True,num_workers=2)
print(f'total train data:{len(train_dl.dataset)}')
print(f'total val data:{len(val_dl.dataset)}')


device='cuda:0' if torch.cuda.is_available() else 'cpu'
net=ViT.ViT()
net=net.to(device)
print(summary(net,(3,224,224),device='cuda'))
criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.Adam(net.parameters(), lr=0.001)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)



train_loss_list=[]
val_loss_list=[]
train_acc_list=[]
val_acc_list=[]
max_epoch=100
best_loss=float('inf')
best_acc=0

for epoch in range(max_epoch):
    print('*'*30)
    current_lr=get_lr(optimizer)
    print(f'{epoch+1}/{max_epoch} current_lr:{current_lr}')
    start_time=time.time()
    
    net.train()
    train_loss, train_metric = loss_epoch(net, criterion, train_dl, phase='train',opt=optimizer)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_metric)

    net.eval()
    with torch.no_grad():
        val_loss, val_metric = loss_epoch(net, criterion, val_dl,phase='val')
    val_loss_list.append(val_loss)
    val_acc_list.append(val_metric)

    if val_loss < best_loss:
        best_loss = val_loss
        best_acc=val_metric
        best_model_wts = copy.deepcopy(net.state_dict())
        torch.save(net.state_dict(), 'experiment/SwinT_STL/best.pt')
        print('Copied best model weights!')

    lr_scheduler.step(val_loss)
    if current_lr != get_lr(optimizer):
        print('Loading best model weights!')
        net.load_state_dict(best_model_wts)

    print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' %(train_loss, val_loss, 100*val_metric, (time.time()-start_time)/60))
    
print(f'best acc:{best_acc} best loss:{best_loss}')

np.save('train_acc',train_acc_list)
np.save('val_acc',val_acc_list)
np.save('train_loss',train_loss_list)
np.save('val_loss',val_loss_list)