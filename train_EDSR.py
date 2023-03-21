import time
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from models.EDSR import EDSR
from torch.utils.data import DataLoader

import torch
from dataset.sr_dataset import DatasetSR
from tqdm import tqdm
from utils.utils import calculate_psnr,calculate_ssim
from einops import rearrange
from torchsummary import summary




def main(scale=0):

    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'

    upscale = 2

    
    train_dataset=DatasetSR(phase='train')
    val_dataset=DatasetSR(phase='val')
    train_dataloader=DataLoader(train_dataset,batch_size=64,num_workers=16,shuffle=True)
    val_dataloader=DataLoader(val_dataset,batch_size=1,num_workers=1,shuffle=True)

   
    model = EDSR(scale=upscale)
    model.to(device)
    
    optimizer=optim.Adam(model.parameters(),lr=10e-4,betas=[0.9,0.999],eps=10e-8)
    criterion=nn.L1Loss()
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100000,200000,300000,400000], gamma=0.5)
    step=0
    best_loss=float('inf')

    for epoch in range(500000): 
        current_lr=optimizer.param_groups[0]['lr']
        print('-'*20)
        print(f'epoch:{epoch+1}, current_lr:{current_lr}')
        model.train()
        train_tq=tqdm(train_dataloader, ncols=80, smoothing=0, bar_format='train: {desc}|{bar}{r_bar}')
        for imgs in train_tq:
            step+=1
            HR_img,LR_img=imgs['H'],imgs['L']
            HR_img=HR_img.to(device)
            LR_img=LR_img.to(device)
            optimizer.zero_grad()
            
            
            output_img=model(LR_img)
            loss=criterion(output_img,HR_img)
            loss.backward()
            optimizer.step()
        scheduler.step()        
        
        if step%1300==0:
            current_loss=0
            psnr=0
            ssim=0
            model.eval()
            val_tq=tqdm(val_dataloader, ncols=80, smoothing=0, bar_format='val: {desc}|{bar}{r_bar}')
            with torch.no_grad():
                for idx,imgs in enumerate(val_tq):
                    if idx==10:
                        break
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
                    
            
            epoch_loss=current_loss / 10
            
            if best_loss > epoch_loss:
                torch.save(model.state_dict(),f'experiment/SwinIR/best.pt')
                
                
            avg_pnsr=psnr / 10
            avg_ssim=ssim / 10
            
            
            print(f'epoch:{epoch+1}, iter:{step}, Average PSNR:{avg_pnsr:.4f}, Average SSIM:{avg_ssim:.4f}, loss:{epoch_loss:.4f}\n')
        

if __name__ == '__main__':
    main()