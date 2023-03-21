'''
@Article{DistgLF,
    author    = {Wang, Yingqian and Wang, Longguang and Wu, Gaochang and Yang, Jungang and An, Wei and Yu, Jingyi and Guo, Yulan},
    title     = {Disentangling Light Fields for Super-Resolution and Disparity Estimation},
    journal   = {IEEE TPAMI},
    year      = {2022},
}
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from einops import rearrange


def LF_rgb2ycbcr(x):
    y = torch.zeros_like(x)
    y[:,0,:,:,:,:] =  65.481 * x[:,0,:,:,:,:] + 128.553 * x[:,1,:,:,:,:] +  24.966 * x[:,2,:,:,:,:] +  16.0
    y[:,1,:,:,:,:] = -37.797 * x[:,0,:,:,:,:] -  74.203 * x[:,1,:,:,:,:] + 112.000 * x[:,2,:,:,:,:] + 128.0
    y[:,2,:,:,:,:] = 112.000 * x[:,0,:,:,:,:] -  93.786 * x[:,1,:,:,:,:] -  18.214 * x[:,2,:,:,:,:] + 128.0

    y = y / 255.0
    return y


def LF_ycbcr2rgb(x):
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.matmul(mat_inv, np.array([16, 128, 128]))
    mat_inv = mat_inv * 255

    y = torch.zeros_like(x)
    y[:,0,:,:,:,:] = mat_inv[0,0] * x[:,0,:,:,:,:] + mat_inv[0,1] * x[:,1,:,:,:,:] + mat_inv[0,2] * x[:,2,:,:,:,:] - offset[0]
    y[:,1,:,:,:,:] = mat_inv[1,0] * x[:,0,:,:,:,:] + mat_inv[1,1] * x[:,1,:,:,:,:] + mat_inv[1,2] * x[:,2,:,:,:,:] - offset[1]
    y[:,2,:,:,:,:] = mat_inv[2,0] * x[:,0,:,:,:,:] + mat_inv[2,1] * x[:,1,:,:,:,:] + mat_inv[2,2] * x[:,2,:,:,:,:] - offset[2]
    return y


def LF_interpolate(LF, scale_factor, mode):
    [b, c, u, v, h, w] = LF.size()
    LF = rearrange(LF, 'b c u v h w -> (b u v) c h w')
    LF_upscale = F.interpolate(LF, scale_factor=scale_factor, mode=mode, align_corners=False)
    LF_upscale = rearrange(LF_upscale, '(b u v) c h w -> b c u v h w', u=u, v=v)
    return LF_upscale

class DistgSSR(nn.Module):
    def __init__(self, angRes_in, scale_factor):
        super(DistgSSR, self).__init__()
        channels = 128
        n_group = 6
        n_block = 6
        self.angRes = angRes_in
        self.factor = scale_factor

        self.conv_init0 = nn.Sequential(nn.Conv3d(1, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False))
        self.conv_init = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.disentg = CascadeDisentgGroup(n_group, n_block, self.angRes, channels)
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * self.factor ** 2, kernel_size=1, padding=0, bias=False),
            nn.PixelShuffle(self.factor),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.1),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=False))

    def forward(self, lr, info=None):

        lr = rearrange(lr, 'b c (u h) (v w) -> b c u v h w', u=self.angRes, v=self.angRes)
        [b, c, u, v, h, w] = lr.size()

        # rgb2ycbcr
        # lr_ycbcr = LF_rgb2ycbcr(lr)
        sr_ycbcr = LF_interpolate(lr, scale_factor=self.factor, mode='bicubic')

        # Initial Feature Extraction
        x = rearrange(lr[:, :, :, :, :, :], 'b c u v h w -> b c (u v) h w')
        buffer = self.conv_init0(x)
        buffer = self.conv_init(buffer) + buffer

        buffer = rearrange(buffer, 'b c (u v) h w -> b c (u h) (v w)', u=self.angRes, v=self.angRes)
        buffer = SAI2MacPI(buffer, self.angRes)
        buffer = self.disentg(buffer)
        buffer_SAI = MacPI2SAI(buffer, self.angRes)

        # UP-Sampling
        out = self.upsample(buffer_SAI)
        out = rearrange(out, 'b c (u h) (v w) -> b c u v h w', u=self.angRes, v=self.angRes)
        
        # ycbcr2rgb
        sr_ycbcr += out

        return rearrange(sr_ycbcr, 'b c u v h w -> b c (u h) (v w)', u=self.angRes, v=self.angRes)


class CascadeDisentgGroup(nn.Module):
    def __init__(self, n_group, n_block, angRes, channels):
        super(CascadeDisentgGroup, self).__init__()
        self.n_group = n_group
        Groups = []
        for i in range(n_group):
            Groups.append(DisentgGroup(n_block, angRes, channels))
        self.Group = nn.Sequential(*Groups)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False)

    def forward(self, x):
        buffer = x
        for i in range(self.n_group):
            buffer = self.Group[i](buffer)
        return self.conv(buffer) + x


class DisentgGroup(nn.Module):
    def __init__(self, n_block, angRes, channels):
        super(DisentgGroup, self).__init__()
        self.n_block = n_block
        Blocks = []
        for i in range(n_block):
            Blocks.append(DisentgBlock(angRes, channels))
        self.Block = nn.Sequential(*Blocks)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False)

    def forward(self, x):
        buffer = x
        for i in range(self.n_block):
            buffer = self.Block[i](buffer)
        return self.conv(buffer) + x


class DisentgBlock(nn.Module):
    def __init__(self, angRes, channels):
        super(DisentgBlock, self).__init__()
        SpaChannel, AngChannel, EpiChannel = channels, channels//4, channels//2

        self.SpaConv = nn.Sequential(
            nn.Conv2d(channels, SpaChannel, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(SpaChannel, SpaChannel, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.AngConv = nn.Sequential(
            nn.Conv2d(channels, AngChannel, kernel_size=angRes, stride=angRes, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(AngChannel, angRes * angRes * AngChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.PixelShuffle(angRes),
        )
        self.EPIConv = nn.Sequential(
            nn.Conv2d(channels, EpiChannel, kernel_size=[1, angRes * angRes], stride=[1, angRes], padding=[0, angRes * (angRes - 1)//2], bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(EpiChannel, angRes * EpiChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            PixelShuffle1D(angRes),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(SpaChannel + AngChannel + 2 * EpiChannel, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
        )

    def forward(self, x):
        feaSpa = self.SpaConv(x)
        feaAng = self.AngConv(x)
        feaEpiH = self.EPIConv(x)
        feaEpiV = self.EPIConv(x.permute(0, 1, 3, 2).contiguous()).permute(0, 1, 3, 2)
        buffer = torch.cat((feaSpa, feaAng, feaEpiH, feaEpiV), dim=1)
        buffer = self.fuse(buffer)
        return buffer + x


class PixelShuffle1D(nn.Module):
    """
    1D pixel shuffler
    Upscales the last dimension (i.e., W) of a tensor by reducing its channel length
    inout: x of size [b, factor*c, h, w]
    output: y of size [b, c, h, w*factor]
    """
    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self.factor = factor

    def forward(self, x):
        b, fc, h, w = x.shape
        c = fc // self.factor
        x = x.contiguous().view(b, self.factor, c, h, w)
        x = x.permute(0, 2, 3, 4, 1).contiguous()           # b, c, h, w, factor
        y = x.view(b, c, h, w * self.factor)
        return y


def MacPI2SAI(x, angRes):
    out = []
    for i in range(angRes):
        out_h = []
        for j in range(angRes):
            out_h.append(x[:, :, i::angRes, j::angRes])
        out.append(torch.cat(out_h, 3))
    out = torch.cat(out, 2)
    return out


def SAI2MacPI(x, angRes):
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])
        tempU.append(torch.cat(tempV, dim=3))
    out = torch.cat(tempU, dim=2)
    return out


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        # self.criterion_Loss = torch.nn.L1Loss()
        self.criterion_Loss = torch.nn.SmoothL1Loss()

    def forward(self, out, HR, degrade_info=None):
        loss = self.criterion_Loss(out['SR'], HR)

        return loss


def weights_init(m):
    pass


