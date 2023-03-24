import numpy as np
import matplotlib.pyplot as plt
from dataset.sr_dataset import DatasetSR
from PIL import Image as im

from models.EDSR import EDSR
import torch
import torch.nn as nn
from utils.utils import calculate_psnr,calculate_ssim
from tqdm import tqdm
from dataset.sr_dataset import DatasetSR
from einops import rearrange
from torch.utils.data import DataLoader
import cv2
import glob
import json

device='cuda:0' if torch.cuda.is_available() else 'cpu'

val_dataset=DatasetSR(phase='val')
val_dataloader=DataLoader(val_dataset,batch_size=1,num_workers=1,shuffle=True)

weights=glob.glob('experiment/EDSR/*.pt')
di={}


for weight_path in weights:

    model=EDSR(scale=2)
    weight=torch.load(weight_path)
    model.load_state_dict(weight)
    model.to(device)


    val_tq=tqdm(val_dataloader, ncols=80, smoothing=0, bar_format='val: {desc}|{bar}{r_bar}')

    current_loss=0
    psnr=0
    ssim=0
    criterion=nn.L1Loss()
    with torch.no_grad():
        for idx,imgs in enumerate(val_tq):
           
        
            HR_img=imgs['H'].to(device)
            LR_img=imgs['L'].to(device)
            
            output_img=model(LR_img)
            current_loss+=criterion(output_img,HR_img)
            
            output_img=output_img[0].cpu().numpy()
            HR_img=HR_img[0].cpu().numpy()
            output_img=(rearrange(output_img,'c h w -> h w c'))
            HR_img=(rearrange(HR_img,'c h w -> h w c'))

            psnr+=calculate_psnr(output_img,HR_img,crop_border=8)
            ssim+=calculate_ssim(output_img,HR_img,crop_border=8)
            
            
            #####################
            # save result image #
            #####################
            
            # output_img=cv2.cvtColor(output_img,cv2.COLOR_BGR2RGB)
            # HR_img=cv2.cvtColor(HR_img,cv2.COLOR_BGR2RGB)
            # LR_img=LR_img[0].cpu().numpy()
            # LR_img=(rearrange(LR_img,'c h w -> h w c'))
            # LR_img=cv2.cvtColor(LR_img,cv2.COLOR_BGR2RGB)
            # LR_img=cv2.resize(LR_img,(output_img.shape[1],output_img.shape[0]),)
            
            # cv2.imwrite(f'SR_{idx}.png',output_img)
            # cv2.imwrite(f'HR_{idx}.png',HR_img)
            # cv2.imwrite(f'LR_{idx}.png',LR_img)
            
            
        epoch_loss=current_loss / 100
        avg_psnr=psnr / 100
        avg_ssim=ssim / 100
        
        print(avg_psnr)
        print(avg_ssim)
        
        
    di[weight_path]=[avg_psnr,avg_ssim,epoch_loss.item()]
    

with open('dict.json','w') as f:
    json.dump(di,f)
f.close
    