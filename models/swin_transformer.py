# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu -> changed -> using einops library  
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


class Mlp(nn.Module):
    def __init__(self,in_features,hidden_features=None,out_features=None,act_layer=nn.GELU,drop=0.):
        super().__init__()
        out_features=out_features or in_features
        hidden_features=hidden_features or in_features
        self.f1=nn.Linear(in_features,hidden_features)
        self.act=act_layer()
        self.fc2=nn.Linear(hidden_features,out_features)
        self.drop=nn.Dropout(drop)
        
    def forward(self,x):
        x=self.fc1(x)
        x=self.act(x)
        x=self.drop(x)
        x=self.f2(x)
        x=self.drop(x)
        return x
    
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    # B, H, W, C = x.shape
    # x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    
    x=rearrange(x,'b (h s1) (w s2) c -> b h s1 w s2 c',s1=window_size,s2=window_size)
    windows=rearrange(x,'b h s1 w s2 c -> (b h w) s1 s2 c')
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    #x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    
    x=rearrange(windows,'(b h w) s1 s2 c -> b h w s1 s2 c',b=B,h=int(H/window_size),w=int(W/window_size),s2=window_size,s1=window_size)
    x=rearrange(x,'b h w s1 s2 c -> b (h s1) (w s2) c')
   
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    def __init__(self,dim,window_size,num_heads,qkv_bias=True,qk_scale=None,attn_drop=0,proj_drop=0):
        super().__init__()
        self.dim=dim
        self.window_size=window_size   # Wh, Ww
        self.num_heads=num_heads
        head_dim=dim//num_heads
        self.scale=qk_scale or head_dim ** -0.5
        
        # define a parameter table of relative position bias
        self.relative_position_bias_table=nn.Parameter(
            torch.zeros((2*window_size[0] - 1) * (2 * window_size[1] -1), num_heads)
        )
        
        # get pair wise relative position index for each token inside the window
        coords_h=torch.arange(self.window_size[0])
        coords_w=torch.arange(self.window_size[1])
        coords=torch.stack(torch.meshgrid([coords_h,coords_w]))  #(2, Wh, Ww)
        coords_flatten=torch.flatten(coords,1)  # (2,49)
        relative_coords=coords_flatten[:,:,None] - coords_flatten[:,None,:]  # (2,Wh,Ww)
        relative_coords=rearrange(relative_coords,'c h w -> h w c')
        relative_coords[:,:,0] += self.window_size[0] - 1      # shift to start from 0
        relative_coords[:,:,1] += self.window_size[1] - 1
        relative_coords[:,:,0] *= 2 * self.window_size[1] - 1
        relative_position_index=relative_coords.sum(-1)    # (Wh,Ww)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv=nn.Linear(dim,dim*3, bias=qkv_bias)
        self.attn_drop=nn.Dropout(attn_drop)
        self.proj=nn.Linear(dim,dim)
        self.proj_drop=nn.Dropout(proj_drop)
        
        trunc_normal_(self.relative_position_bias_table,std=.02)
        self.softmax=nn.Softmax(dim=-1)
        
    
    def forward(self,x,mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_,N,C=x.shape
        
        
        
    







class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    
    def __init__(self,img_size=224,patch_size=4,in_chans=3,embed_dim=96,norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)  #(224,224)
        patch_size = to_2tuple(patch_size)  #(4,4)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # [56,56]
        self.img_size=img_size
        self.patch_size=patch_size
        self.patches_resolution=patches_resolution
        self.num_patches=patches_resolution[0] * patches_resolution[1]  # [56 * 56]
        
        self.in_chans=in_chans
        self.embed_dim=embed_dim
        
        self.proj=nn.Conv2d(in_chans,embed_dim,kernel_size=patch_size,stride=patch_size) #(3,224,224) -> (96,56,56)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
    
    def forward(self,x):
        B,C,H,W=x.shape  #(B,3,224,224)
        assert H==self.img_size[0] and W==self.img_size[1],\
            f"Input image size ({H}x{W}) dosen't match model ({self.img_size[0]} *{self.img_size[1]})"
            
        x=rearrange(self.proj(x),'B C H W -> B (H W) C')  # B Ph*Pw C   (B,96,56,56) -> (B, 56*56, 96)
        if self.norm is not None:
            x=self.norm(x)
        return x
    
    def flops(self):
        Ho,Wo = self.patches_resolution
        flops=Ho*Wo*self.embed_dim*self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
    






if __name__=='__main__':
    device='cuda:0'
    input=torch.randn((16,3,224,224)).to(device)
    
    # windows=window_partition(input,7)
    # print(windows.shape)
    # x=window_reverse(windows,7,224,224)
    # print(x.shape)
    
    # model=WindowAttention()
    
    patch_embed=PatchEmbed().to(device)
    embed_patch=patch_embed(input)
    print(embed_patch.shape)
    