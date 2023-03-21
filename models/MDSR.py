import torch.nn as nn
import torch
import math
import common
'''
MDSR
depth(B): the number of layers = 16
width(F): the number of feature channels = 64 or 256
scaling factor = 0.1
x3, x4배 모델을 학습시킬때 x2 모델로 학습된 모델을 사용함


'''
        
        
class MDSR(nn.Module):
    def __init__(self,conv=common.default_conv):
        super(MDSR,self).__init__()
        n_resblocks=16
        n_feats=64
        kernel_size=3
        act=nn.ReLU(True)
        scale=[2,3,4]
        self.scale_idx=0
        self.sub_mean=common.MeanShift(255)
        self.add_mean=common.MeanShift(255,sign=1)
        
        m_head=[conv(3,n_feats,kernel_size=kernel_size)]
        
        self.pre_process=nn.ModuleList([
            nn.Sequential(
                common.ResBlock(conv,n_feats,5,act=act),
                common.ResBlock(conv,n_feats,5,act=act)
            ) for _ in scale
        ])
        
        print(self.pre_process)
        m_body=[
            common.ResBlock(
                conv,n_feats,kernel_size,act
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats,n_feats,kernel_size))
        
        self.upsample=nn.ModuleList([
            common.Upsampler(conv,s,n_feats,act=False) for s in scale
        ])
        
        m_tail=[conv(n_feats,3,kernel_size)]
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)


    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        x = self.pre_process[self.scale_idx](x)

        res = self.body(x)
        res += x

        x = self.upsample[self.scale_idx](res)
        x = self.tail(x)
        x = self.add_mean(x)

        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx


if __name__ == '__main__':
    model=MDSR()
    model.set_scale(2)
    x=torch.rand((3,24,24))
    x=model(x)
    print(x.shape)