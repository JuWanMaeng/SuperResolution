import time
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from model.network_swinir import SwinIR
from torch.utils.data import DataLoader

import torch
from dataset.dataset import DIV2K_dataset
from tqdm import tqdm
from utils.utils import calculate_psnr,calculate_ssim




def main(scale):

    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'

  
    train_dataset=DIV2K_dataset(phase='train')
    val_dataset=DIV2K_dataset(phase='val')
    train_dataloader=DataLoader(train_dataset,batch_size=32,num_workers=8,shuffle=True)
    val_dataloader=DataLoader(val_dataset,batch_size=1,num_workers=1,shuffle=True)

   
    model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
    model.to(device)
    optimizer=optim.Adam(model.parameters(),lr=2e-4)
    criterion=nn.L1Loss()
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[250000, 400000, 450000, 475000, 500000], gamma=0.5)
    step=0

   

    for epoch in range(1000000):  # keep running
        current_lr=optimizer.param_groups[0]['lr']
        train_tq=tqdm(train_dataloader, ncols=80, smoothing=0, bar_format='train: {desc}|{bar}{r_bar}')
        print('-'*20)
        print(f'epoch:{epoch}, current_lr:{current_lr}')
        model.train()
        
        for HR_img,LR_img in train_tq:
            step+=1
            HR_img=HR_img.to(device)
            LR_img=LR_img.to(device)
            optimizer.zero_grad()
            
            
            output_img=model(LR_img)
            loss=criterion(output_img,HR_img)
            loss.backward()
            optimizer.step()
        scheduler.step()        
        
        if step%5000==0:
            psnr=0
            ssim=0
            model.eval()
            val_tq=tqdm(val_dataloader, ncols=80, smoothing=0, bar_format='val: {desc}|{bar}{r_bar}')
            with torch.no_grad():
                for HR_img,LR_img in val_tq:
                    HR_img=HR_img.cpu().numpy()
                    LR_img=LR_img.to(device)
                    output_img=model(LR_img).cpu().numpy()
                    loss=criterion(output_img,HR_img)
                    psnr+=calculate_psnr(output_img,HR_img,border=scale)
                    ssim+=calculate_ssim(output_img,HR_img,border=scale)
                    
            avg_pnsr=psnr / len(val_dataloader.dataset)
            avg_ssim=ssim / len(val_dataloader.dataset)
            
            
            print(f'epoch:{epoch}, iter:{step}, Average PSNR:{avg_pnsr:.4f}, Average SSIM:{avg_ssim:.4f}\n')
            
            
            
                    
                    
                    
                    
            
            

if __name__ == '__main__':
    main(scale=4)