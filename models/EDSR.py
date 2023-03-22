import torch.nn as nn
import torch
import math
'''
EDSR
https://github.com/sanghyun-son/EDSR-PyTorch
depth(B): the number of layers = 32
width(F): the number of feature channels = 64 or 256
scaling factor = 0.1
x3, x4배 모델을 학습시킬때 x2 모델로 학습된 모델을 사용함


'''
    
class MeanShift(nn.Conv2d):  # 입력 이미지 데이터에서 평균 값을 빼고, 표준 편차를 나누어 이미지 데이터를 정규화함
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1): # sign: 이미지 밝기를 조정

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False



class ResBlock(nn.Module):
    def __init__(
        self,n_feats,kernel_size,bias=True,bn=False,
        act=nn.ReLU(True),res_scale=1):
        super(ResBlock,self).__init__()
        m=[]
        for i in range(2):
            m.append(nn.Conv2d(n_feats,n_feats,kernel_size,padding=kernel_size//2,bias=bias))
            if i==0:
                m.append(act)
                
        self.body=nn.Sequential(*m)
        self.res_scale=res_scale   # 1 or 0.1
        
    def forward(self,x):
        res=self.body(x).mul(self.res_scale)
        res+=x
        
        return x

class Upsampler(nn.Sequential):
    def __init__(self,scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feats, 4 * n_feats, 3,padding=3//2,bias=True))
                m.append(nn.PixelShuffle(2))
                
        elif scale == 3:
            m.append(nn.Conv2d(n_feats, 9 * n_feats, 3,padding=3//2, bias=True))
            m.append(nn.PixelShuffle(3))
        
        super(Upsampler, self).__init__(*m)





class EDSR(nn.Module):
    def __init__(self,scale):
        super(EDSR,self).__init__()
        
        n_resblocks=32
        n_feats=64
        kernel_size=3
        scale=scale
        res_scale=1
        
        act=nn.ReLU(True)
        
        self.sub_mean=MeanShift(rgb_range=255)
        self.add_mean=MeanShift(rgb_range=255,sign=1)
        
        # define head module
        m_head=[nn.Conv2d(3,n_feats,kernel_size=kernel_size,padding=kernel_size//2,bias=True)]
        
        # define body module
        m_body=[
            ResBlock(
                n_feats,kernel_size=3,act=act,res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        
        # define tail module
        m_tail=[
            Upsampler(scale,n_feats,act=False),
            nn.Conv2d(n_feats,3,kernel_size=kernel_size,padding=kernel_size//2,bias=True)
        ]
    
    
        self.head=nn.Sequential(*m_head)
        self.body=nn.Sequential(*m_body)
        self.tail=nn.Sequential(*m_tail)
        
    def forward(self,x):
        x=self.sub_mean(x)
        x=self.head(x)
        
        res=self.body(x)
        res+=x
        
        x=self.tail(res)
        x=self.add_mean(x)

        return x
        
        
    

    
