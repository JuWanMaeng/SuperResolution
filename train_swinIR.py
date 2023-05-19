import time
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

from SwinIR import SwinIR
from models.EDSR import EDSR
from torch.utils.data import DataLoader

import torch
from dataset.sr_dataset import DatasetSR
from tqdm import tqdm
from utils.utils import calculate_psnr,calculate_ssim
from einops import rearrange
from torchsummary import summary
import wandb




def main(scale,device):
    wandb.init(project='SwinIR',entity='aodwndhks')
    wandb.run.name=(f'SwinIR_x{scale}')


    upscale = scale
    training_patch_size=128
    batch_size=4
    
    iter_per_epoch=800//batch_size
    max_epoch=500000//iter_per_epoch
    
    milestones=[max_epoch//2, max_epoch*3//5, max_epoch*4//5, max_epoch*9//10]
    print(f'max_epoch:{max_epoch}, milestones:{milestones}')
    
    
    train_dataset=DatasetSR(phase='train',scale=upscale)
    val_dataset=DatasetSR(phase='val',scale=upscale)
    train_dataloader=DataLoader(train_dataset,batch_size=batch_size,num_workers=4,shuffle=True)
    val_dataloader=DataLoader(val_dataset,batch_size=1,num_workers=1,shuffle=True)

   
    model = SwinIR(upscale=upscale, in_chans=3, img_size=training_patch_size, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    model.to(device)
    
    optimizer=optim.Adam(model.parameters(),lr=2e-4)
    criterion=nn.L1Loss()
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    step=0

    best_loss=float('inf')


    for epoch in range(max_epoch):  # keep running
        current_lr=optimizer.param_groups[0]['lr']
        print('-'*20)
        print(f'epoch:{epoch+1}, current_lr:{current_lr}')
        model.train()
        train_tq=tqdm(train_dataloader, ncols=80, smoothing=0, bar_format='train: {desc}|{bar}{r_bar}')
        
        train_loss=0
        for imgs in train_tq:
            step+=1
            HR_img,LR_img=imgs['H'],imgs['L']
            HR_img=HR_img.to(device)
            LR_img=LR_img.to(device)
            optimizer.zero_grad()
            
            
            output_img=model(LR_img)
            if upscale==3:
                output_img = F.pad(output_img, (0, 2, 0, 2), 'reflect')
            loss=criterion(output_img,HR_img)
            train_loss+=loss.item()
            loss.backward()
            optimizer.step()
            
       
            
        scheduler.step()
        train_loss=train_loss/len(train_dataloader)

        wandb.log({'train_loss':train_loss},step=step)
        
        if (epoch+1)%20==0:
            current_loss=0
            psnr=0
            ssim=0
            model.eval()
            val_tq=tqdm(val_dataloader, ncols=80, smoothing=0, bar_format='val: {desc}|{bar}{r_bar}')
            with torch.no_grad():
                for idx,imgs in enumerate(val_tq):
                    HR_img=imgs['H'].to(device)
                    LR_img=imgs['L'].to(device)
                    
                    output_img=model(LR_img)
                    current_loss+=criterion(output_img,HR_img) 
                    
                    output_img=output_img[0].cpu().numpy()
                    HR_img=HR_img[0].cpu().numpy()
                    output_img=(rearrange(output_img,'c h w -> h w c')*255)
                    HR_img=(rearrange(HR_img,'c h w -> h w c')*255)

                    psnr+=calculate_psnr(output_img,HR_img,crop_border=0)
                    ssim+=calculate_ssim(output_img,HR_img,crop_border=0)
                    
            
            epoch_loss=current_loss / len(val_dataloader)   
            avg_pnsr=psnr / len(val_dataloader)
            avg_ssim=ssim / len(val_dataloader)
            
                        
            if best_loss > epoch_loss:
                best_psnr=avg_pnsr
                best_ssim=avg_ssim
                torch.save(model.state_dict(),f'weights/SwinIR/x{scale}_best.pt')
            
            
            print(f'epoch:{epoch+1}, iter:{step}, Average PSNR:{avg_pnsr:.4f}, Average SSIM:{avg_ssim:.4f}, loss:{epoch_loss:.4f}\n')
            
            wandb.log({'PSNR':avg_pnsr,
                       'SSIM':ssim,
                       'val_loss':epoch_loss},step=step)
            
    wandb.log({'best psnr':best_psnr,
               'best ssim':best_ssim})
            
            
            
                    
         

if __name__ == '__main__':
    if torch.cuda.is_available():
        device='cuda:1'
    else:
        device='cpu'
    main(scale=3,device=device)
    
