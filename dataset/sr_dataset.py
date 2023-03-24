import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms,models
import json
from PIL import Image as im
import cv2
import numpy as np
from torch.utils.data import DataLoader

import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util


class DatasetSR(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, phase):
        super(DatasetSR, self).__init__()
        self.phase = phase
        self.n_channels =  3
        self.sf =  2
        self.patch_size = 96
        self.L_size = self.patch_size // self.sf

        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        
        with open(f'dataset/{phase}_HR.json', 'r') as f:
            self.imgs_H=json.load(f)
        with open(f'dataset/{phase}_x2_LR.json','r') as f:
            self.imgs_L=json.load(f)

        assert self.imgs_H, 'Error: H imgs are empty.'
        if self.imgs_H and self.imgs_L:
            assert len(self.imgs_L) == len(self.imgs_H), 'L/H mismatch - {}, {}.'.format(len(self.imgs_L), len(self.imgs_H))

    def __getitem__(self, index):
        
        
        # ------------------------------------
        # get H image
        # ------------------------------------
        img_H=cv2.imread(self.imgs_H[index]['img'])
        img_H=cv2.cvtColor(img_H,cv2.COLOR_BGR2RGB)
        # img_H=util.uint2single(img_H)
        
        # ------------------------------------
        # modcrop
        # ------------------------------------
        img_H = util.modcrop(img_H, self.sf)
        # ------------------------------------
        # get L image
        # ------------------------------------
        if self.imgs_L:
            # --------------------------------
            # directly load L image
            # --------------------------------
            img_L=cv2.imread(self.imgs_L[index]['img'])
            img_L=cv2.cvtColor(img_L,cv2.COLOR_BGR2RGB)
            # img_L=util.uint2single(img_L)
            

        else:
            # --------------------------------
            # sythesize L image via matlab's bicubic
            # --------------------------------
            H, W = img_H.shape[:2]
            img_L = util.imresize_np(img_H, 1 / self.sf, True)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.phase== 'train':

            H, W, C = img_L.shape

            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]
          
            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)
        
        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
        

        return {'L': img_L, 'H': img_H}

    def __len__(self):
        return len(self.imgs_H)





if __name__ == '__main__':
    tt=DatasetSR(phase='train')
    train_dataloader=DataLoader(tt,batch_size=32,num_workers=8,shuffle=True)
    for i in range(len(train_dataloader.dataset)):
        print(tt[i][0].shape)
        print(type(tt[i][0]))
   
