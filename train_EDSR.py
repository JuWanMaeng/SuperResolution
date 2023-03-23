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
import copy


# get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']



def main(scale=0):

    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'

    upscale = 2

    
    train_dataset=DatasetSR(phase='train')
    val_dataset=DatasetSR(phase='val')
    train_dataloader=DataLoader(train_dataset,batch_size=16,num_workers=8,shuffle=True)
    val_dataloader=DataLoader(val_dataset,batch_size=1,num_workers=1,shuffle=True)

   
    model = EDSR(scale=upscale)
    model.to(device)
    
    optimizer=optim.Adam(model.parameters(),lr=10e-4,betas=[0.9,0.999],eps=10e-8)
    criterion=nn.L1Loss()
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[180,360,540,720],gamma=0.5)
    step=0
    best_loss=float('inf')
    best_psnr=0
    

    for epoch in range(1,801): 
        current_lr=optimizer.param_groups[0]['lr']
        print('-'*60)
        print(f'epoch:{epoch}, current_lr:{current_lr}, iter:{step}')
        
        model.train()
        train_tq=tqdm(train_dataloader, ncols=80, smoothing=0, bar_format='train: {desc}|{bar}{r_bar}')
        for imgs in train_tq:
            step+=16                            # 이미지 한장을 1 step이라고 하자
            HR_img,LR_img=imgs['H'],imgs['L']
            HR_img=HR_img.to(device)
            LR_img=LR_img.to(device)
            optimizer.zero_grad()
            
            
            output_img=model(LR_img)
            loss=criterion(output_img,HR_img)
            loss.backward()
            optimizer.step()        
        
        
        current_loss=0
        psnr=0
        ssim=0
        model.eval()
        val_tq=tqdm(val_dataloader, ncols=80, smoothing=0, bar_format='val: {desc}|{bar}{r_bar}')
        with torch.no_grad():
            for idx,imgs in enumerate(val_tq):
                if idx==10:                   # val은 10장만
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
        avg_psnr=psnr / 10
        avg_ssim=ssim / 10
        
        if epoch % 20==0:
            torch.save(model.state_dict(),f'experiment/EDSR/{step}_{psnr}_{loss}_.pt')
            print('save step weights!')
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'experiment/EDSR/loss_best.pt')
            print('Copied best loss model weights!')
        
        if best_psnr < avg_psnr:
            best_psnr= avg_psnr
            best_psnr_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'experiment/EDSR/psnr_best.pt')
            print('Copied best psnr model weights!')
        
        scheduler.step()
        if current_lr != get_lr(optimizer):   # lr이 감소하게 되면 이전 lr에서 학습된 모델들중 loss가 제일 작았던 모델을 불러온다.
            print('Loading best model weights!')
            model.load_state_dict(best_model_wts)
                
        
        
        
        print(f'epoch:{epoch}, iter:{step}, Average PSNR:{avg_psnr:.4f}, Average SSIM:{avg_ssim:.4f}, loss:{epoch_loss:.4f}\n')
        
        
        psnr_list=np.load('psnr.npy').tolist()
        psnr_list.append(avg_psnr)
        np.save('psnr',psnr_list)
            
            

if __name__ == '__main__':
    psnr_list=[]
    np.save('psnr',psnr_list)
    main()