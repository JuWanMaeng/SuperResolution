import time
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from models.SwinIR import SwinIR
from torch.utils.data import DataLoader

import torch
from dataset.sr_dataset import DatasetSR
from tqdm import tqdm
from utils.utils import calculate_psnr,calculate_ssim
from einops import rearrange




def main(scale=0):

    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'

    upscale = 4
    window_size=8
    height = (1024 // upscale // window_size + 1) * window_size
    width = (720 // upscale // window_size + 1) * window_size
    
    train_dataset=DatasetSR(phase='train')
    val_dataset=DatasetSR(phase='val')
    train_dataloader=DataLoader(train_dataset,batch_size=4,num_workers=8,shuffle=True)
    val_dataloader=DataLoader(val_dataset,batch_size=1,num_workers=1,shuffle=True)

   
    model = SwinIR(upscale=4, in_chans=3, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    model.to(device)
    
    optimizer=optim.Adam(model.parameters(),lr=2e-4)
    criterion=nn.L1Loss()
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[125000, 200000, 225000, 237500, 250000], gamma=0.5)
    step=0

    best_loss=float('inf')

    for epoch in range(500000):  # keep running
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
        
        if step%2000==0:
            current_loss=0
            psnr=0
            ssim=0
            model.eval()
            val_tq=tqdm(val_dataloader, ncols=80, smoothing=0, bar_format='val: {desc}|{bar}{r_bar}')
            with torch.no_grad():
                for imgs in val_tq:
                    HR_img=imgs['H'].to(device)
                    LR_img=imgs['L'].to(device)
                    
                    output_img=model(LR_img)
                    current_loss+=criterion(output_img,HR_img)
                    
                    
                    
                    output_img=output_img[0].cpu().numpy()
                    HR_img=HR_img[0].cpu().numpy()
                    output_img=(rearrange(output_img,'c h w -> h w c')*255).astype(np.uint8)
                    HR_img=(rearrange(HR_img,'c h w -> h w c')*255).astype(np.uint8)

                    psnr+=calculate_psnr(output_img,HR_img,crop_border=0)
                    ssim+=calculate_ssim(output_img,HR_img,crop_border=0)
                    
            
            epoch_loss=current_loss / len(val_dataloader.dataset)
            
            if best_loss > epoch_loss:
                torch.save(model.state_dict(),f'experiment/SwinIR/best.pt')
                
                
            avg_pnsr=psnr / len(val_dataloader.dataset)
            avg_ssim=ssim / len(val_dataloader.dataset)
            
            
            print(f'epoch:{epoch}, iter:{step}, Average PSNR:{avg_pnsr:.4f}, Average SSIM:{avg_ssim:.4f}, loss:{epoch_loss:.4f}\n')
        
        
            
            
            
                    
                    
                    
                    
            
            

if __name__ == '__main__':
    main()
    
    
