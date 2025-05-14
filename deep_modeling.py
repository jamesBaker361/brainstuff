#shamelessly plaigarized from https://github.com/ozcelikfu/brain-diffuser/tree/main/vdvae

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_conv(in_dim, out_dim, kernel_size, stride, padding, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c


def get_3x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups, scaled=scaled)


def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups, scaled=scaled)
class PixelBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels,  residual=False, use_3x3=True, zero_last=False):
        super().__init__()
        self.residual = residual
        self.c1 = get_1x1(in_channels, middle_channels)
        self.c2 = get_3x3(middle_channels, middle_channels) if use_3x3 else get_1x1(middle_channels, middle_channels)
        self.c3 = get_3x3(middle_channels, middle_channels) if use_3x3 else get_1x1(middle_channels, middle_channels)
        self.c4 = get_1x1(middle_channels, out_channels, zero_weights=zero_last)
        self.in_channels=in_channels
        self.middle_channels=middle_channels
        self.out_channels=out_channels
        self.module_list=nn.ModuleList([self.c1,self.c2,self.c3,self.c4])

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        return out
    
class ArrayBlock(nn.Module):
    def __init__(self,in_features,middle_features,out_features,residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.residual=residual
        self.c1=nn.Linear(in_features,middle_features)
        self.c2=nn.Linear(middle_features,middle_features)
        self.c3=nn.Linear(middle_features,middle_features)
        self.c4=nn.Linear(middle_features,out_features)
        self.module_list=nn.ModuleList([self.c1,self.c2,self.c3,self.c4])

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        return out
    
