import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms,models
import json
from PIL import Image as im
import cv2
import numpy as np
from torch.utils.data import DataLoader

class DIV2K_dataset(Dataset):
    def __init__(self,phase):
        super(DIV2K_dataset, self).__init__()
        with open(f'dataset/{phase}_HR.json', 'r') as f:
            self.HR=json.load(f)
        with open(f'dataset/{phase}_LR.json','r') as f:
            self.LR=json.load(f)
        self.phase=phase
        upscale = 4
        window_size = 8
        height = (1024 // upscale // window_size + 1) * window_size
        width = (720 // upscale // window_size + 1) * window_size
        
        self.HR_transform=transforms.Compose([
        
        
            transforms.ToTensor()
        ])
        
        self.LR_transform=transforms.Compose([
            transforms.Resize((1024,1024)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.LR)

    def __getitem__(self, idx):
        HR_path, LR_path=self.HR[idx]['img'],self.LR[idx]['img']
        HR_img,LR_img=im.open(HR_path).convert('RGB'),im.open(LR_path).convert('RGB')
       
        
        if self.phase=='train':
            HR_img,LR_img=self.train_transform(HR_img),self.train_transform(LR_img)
        else:
            HR_img,LR_img=self.val_transform(HR_img),self.val_transform(LR_img)
        
        return HR_img,LR_img




if __name__ == '__main__':
    tt=DIV2K_dataset(phase='train')
    train_dataloader=DataLoader(tt,batch_size=32,num_workers=8,shuffle=True)
    for i in range(len(train_dataloader.dataset)):
        print(tt[i][0].shape)
        print(type(tt[i][0]))
   
