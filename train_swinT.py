import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
import torch.nn as nn
from models import swin_transformer,ViT
import torch.optim as optim
from tqdm import tqdm
import numpy as np



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
val_dl = DataLoader(val_ds, batch_size=32, shuffle=True,num_workers=2)
print(f'total train data:{len(train_dl.dataset)}')
print(f'total val data:{len(val_dl.dataset)}')


device='cuda:0'
net=ViT.ViT()
net=net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)



train_loss=[]
val_loss=[]
train_acc=[]
val_acc=[]
max_epoch=100
best_acc=0
best_epoch=0

for epoch in range(max_epoch):
    print('*'*30)
    print(f'{epoch+1}/{max_epoch}')
    
    tq=tqdm(train_dl, ncols=80, smoothing=0, bar_format='train: {desc}|{bar}{r_bar}')
    
    
    
    running_loss = 0.0
    corrects=0
    net.train()
    for i, data in tqdm(enumerate(tq)):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data
        inputs=inputs.to(device)
        labels=labels.to(device)
       

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        output = net(inputs)
        loss = criterion(output, labels)
        pred=output.argmax(1,keepdim=True)
        corrects+=pred.eq(labels.view_as(pred)).sum().item()
        
        loss.backward()
        optimizer.step()

        # 통계를 출력합니다.
        
        running_loss += loss.item() * inputs.size(0)
    
    t_loss=running_loss/len(train_dl.dataset)
    t_acc=corrects/len(train_dl.dataset)
    train_loss.append(t_loss)
    train_acc.append(t_acc)
    
    print(f'train_loss: {t_loss:.5f}, train_accuray: {t_acc*100:.2f}%')
    
    
    
    running_loss = 0.0
    corrects=0
    net.eval()
    vtq=tqdm(val_dl, ncols=80, smoothing=0, bar_format='val: {desc}|{bar}{r_bar}')
    for i, data in tqdm(enumerate(vtq)):
        inputs, labels = data
        inputs=inputs.to(device)
        labels=labels.to(device)
       

        with torch.no_grad():
            output = net(inputs)
            loss = criterion(output, labels)
            pred=output.argmax(1,keepdim=True)
            corrects+=pred.eq(labels.view_as(pred)).sum().item()
            running_loss += loss.item() * inputs.size(0)
        
    
    v_loss=running_loss/len(val_dl.dataset)
    v_acc=corrects/len(val_dl.dataset)
    
    if v_acc>best_acc:
        best_acc=v_acc
        best_epoch=epoch
        torch.save(net.state_dict(),'experiment/SwinT_STL/best.pt')
        print('best model saved')
    
    
    val_loss.append(v_loss)
    val_acc.append(v_acc)
    
    print(f'val_loss: {v_loss:.5f}, val_accuracy{v_acc*100:.2f}%')
    

print(f'best acc:{best_acc} at epoch:{best_epoch}')

np.save('train_acc',train_acc)
np.save('val_acc',val_acc)
np.save('train_loss',train_loss)
np.save('val_loss',val_loss)