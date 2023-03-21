# Copyright 2023 The KaiJIN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""https://github.com/ZhengyuLiang24/EPIT

@Article{EPIT,
    author    = {Liang, Zhengyu and Wang, Yingqian and Wang, Longguang and Yang, Jungang and Zhou Shilin and Guo, Yulan},
    title     = {Learning Non-Local Spatial-Angular Correlation for Light Field Image Super-Resolution},
    journal   = {arXiv preprint arXiv:}, 
    year      = {2023},   
}

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class DistgEPIT_deeper(nn.Module):
  def __init__(self, angRes_in, scale_factor):
    super(DistgEPIT_deeper, self).__init__()
    channels = 64
    self.angRes = angRes_in
    self.scale = scale_factor

    #################### Initial Feature Extraction #####################
    self.conv_init0 = nn.Sequential(nn.Conv3d(1, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False))
    self.conv_init = nn.Sequential(
        nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        nn.LeakyReLU(0.2, inplace=True),
    )

    ############# Deep Spatial-Angular Correlation Learning #############
    self.altblock = nn.Sequential(
        AltFilter(self.angRes, channels),
        AltFilter(self.angRes, channels),
        AltFilter(self.angRes, channels),
        AltFilter(self.angRes, channels),
        AltFilter(self.angRes, channels),
        AltFilter(self.angRes, channels),
        AltFilter(self.angRes, channels),
        AltFilter(self.angRes, channels),
    )

    ########################### UP-Sampling #############################
    self.upsampling = nn.Sequential(
        nn.Conv2d(channels, channels * self.scale ** 2, kernel_size=1, padding=0, bias=False),
        nn.PixelShuffle(self.scale),
        nn.LeakyReLU(0.2),
        nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=False),
    )

    ########################### DistgSSR #############################
    n_group = 8
    n_block = 4
    self.init_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                               dilation=self.angRes, padding=self.angRes, bias=False)
    self.disentg = CascadeDisentgGroup(n_group, n_block, self.angRes, channels)
    self.upsample = nn.Sequential(
        nn.Conv2d(channels, channels * self.scale ** 2, kernel_size=1, stride=1, padding=0),
        nn.PixelShuffle(self.scale),
        nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False))

    ########################### Weight #############################
    # self.w1 = nn.Parameter(torch.tensor(0.5))
    # self.w2 = nn.Parameter(torch.tensor(0.5))

  def forward_distg(self, x, x_feat):
    # b c (u h) (v w) -> b c (h u) (w v)
    buffer = SAI2MacPI(x, self.angRes)
    buffer = self.init_conv(buffer) + x_feat
    buffer = self.disentg(buffer) + x_feat
    # b c (h u) (w v) -> b c (u h) (v w)
    buffer_SAI = MacPI2SAI(buffer, self.angRes)
    out = self.upsample(buffer_SAI)
    return out

  def forward(self, lr, info=None):

    sr = F.interpolate(lr, scale_factor=self.scale, mode='bicubic', align_corners=False)
    x = rearrange(lr, 'b c (u h) (v w) -> b c (u v) h w', u=self.angRes, v=self.angRes)

    # b c (u v) h w
    buffer = self.conv_init0(x)
    buffer = self.conv_init(buffer) + buffer

    distg_x = rearrange(buffer, 'b c (u v) h w -> b c (u h) (v w)', u=self.angRes, v=self.angRes)

    # Deep Spatial-Angular Correlation Learning
    buffer = self.altblock(buffer) + buffer

    # UP-Sampling
    buffer = rearrange(buffer, 'b c (u v) h w -> b c (u h) (v w)', u=self.angRes, v=self.angRes)
    y = self.upsampling(buffer)

    # ssr
    y_1 = self.forward_distg(distg_x, buffer)

    sr = 0.5 * y + 0.5 * y_1 + sr

    return sr


class BasicTrans(nn.Module):
  def __init__(self, channels, spa_dim, num_heads=8, dropout=0.):
    super(BasicTrans, self).__init__()
    self.linear_in = nn.Linear(channels, spa_dim, bias=False)
    self.norm = nn.LayerNorm(spa_dim)
    self.attention = nn.MultiheadAttention(spa_dim, num_heads, dropout, bias=False)
    nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
    self.attention.out_proj.bias = None
    self.attention.in_proj_bias = None
    self.feed_forward = nn.Sequential(
        nn.LayerNorm(spa_dim),
        nn.Linear(spa_dim, spa_dim*2, bias=False),
        nn.ReLU(True),
        nn.Dropout(dropout),
        nn.Linear(spa_dim*2, spa_dim, bias=False),
        nn.Dropout(dropout)
    )
    self.linear_out = nn.Linear(spa_dim, channels, bias=False)

  def gen_mask(self, h: int, w: int, k_h: int, k_w: int):
    attn_mask = torch.zeros([h, w, h, w])
    k_h_left = k_h // 2
    k_h_right = k_h - k_h_left
    k_w_left = k_w // 2
    k_w_right = k_w - k_w_left
    for i in range(h):
      for j in range(w):
        temp = torch.zeros(h, w)
        temp[max(0, i - k_h_left):min(h, i + k_h_right), max(0, j - k_w_left):min(w, j + k_w_right)] = 1
        attn_mask[i, j, :, :] = temp

    attn_mask = rearrange(attn_mask, 'a b c d -> (a b) (c d)')
    attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

    return attn_mask

  def forward(self, buffer):
    [_, _, n, v, w] = buffer.size()
    attn_mask = self.gen_mask(v, w, self.mask_field[0], self.mask_field[1]).to(buffer.device)

    epi_token = rearrange(buffer, 'b c n v w -> (v w) (b n) c')
    epi_token = self.linear_in(epi_token)

    epi_token_norm = self.norm(epi_token)
    epi_token = self.attention(query=epi_token_norm,
                               key=epi_token_norm,
                               value=epi_token,
                               attn_mask=attn_mask,
                               need_weights=False)[0] + epi_token

    epi_token = self.feed_forward(epi_token) + epi_token
    epi_token = self.linear_out(epi_token)
    buffer = rearrange(epi_token, '(v w) (b n) c -> b c n v w', v=v, w=w, n=n)

    return buffer


class AltFilter(nn.Module):
  def __init__(self, angRes, channels):
    super(AltFilter, self).__init__()
    self.angRes = angRes
    self.epi_trans = BasicTrans(channels, channels*2)
    self.conv = nn.Sequential(
        nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
    )

  def forward(self, buffer):
    shortcut = buffer
    [_, _, _, h, w] = buffer.size()
    self.epi_trans.mask_field = [self.angRes * 2, 11]

    # Horizontal
    buffer = rearrange(buffer, 'b c (u v) h w -> b c (v w) u h', u=self.angRes, v=self.angRes)
    buffer = self.epi_trans(buffer)
    buffer = rearrange(buffer, 'b c (v w) u h -> b c (u v) h w', u=self.angRes, v=self.angRes, h=h, w=w)
    buffer = self.conv(buffer) + shortcut

    # Vertical
    buffer = rearrange(buffer, 'b c (u v) h w -> b c (u h) v w', u=self.angRes, v=self.angRes)
    buffer = self.epi_trans(buffer)
    buffer = rearrange(buffer, 'b c (u h) v w -> b c (u v) h w', u=self.angRes, v=self.angRes, h=h, w=w)
    buffer = self.conv(buffer) + shortcut

    return buffer


class CascadeDisentgGroup(nn.Module):
  def __init__(self, n_group, n_block, angRes, channels):
    super(CascadeDisentgGroup, self).__init__()
    self.n_group = n_group
    Groups = []
    for i in range(n_group):
      Groups.append(DisentgGroup(n_block, angRes, channels))
    self.Group = nn.Sequential(*Groups)
    self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                          dilation=int(angRes), padding=int(angRes), bias=False)

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
    self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                          dilation=int(angRes), padding=int(angRes), bias=False)

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
        nn.Conv2d(SpaChannel, SpaChannel, kernel_size=3, stride=1,
                  dilation=int(angRes), padding=int(angRes), bias=False),
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
        nn.Conv2d(channels, EpiChannel, kernel_size=[1, angRes * angRes],
                  stride=[1, angRes], padding=[0, angRes * (angRes - 1)//2], bias=False),
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
  return rearrange(x, 'b c (h u) (w v) -> b c (u h) (v w)', u=angRes, v=angRes)


def SAI2MacPI(x, angRes):
  return rearrange(x, 'b c (u h) (v w) -> b c (h u) (w v)', u=angRes, v=angRes)


if __name__ == "__main__":
  model = DistgEPIT(5, 5)
  model.eval()
  model.to('cuda:0')
  inp = torch.rand(1, 1, 160, 160).to('cuda:0')

  with torch.no_grad():
    out = model(inp)

  print(out.shape)

#   import h5py
#   import numpy as np
#   import imageio
#   import einops

#   def ycbcr2rgb(x):
#     x = x / 255.0
#     mat = np.array(
#         [[65.481, 128.553, 24.966],
#         [-37.797, -74.203, 112.0],
#         [112.0, -93.786, -18.214]])
#     mat_inv = np.linalg.inv(mat)
#     offset = np.matmul(mat_inv, np.array([16, 128, 128]))
#     mat_inv = mat_inv * 255

#     y = np.zeros(x.shape, dtype='double')
#     y[:, :, 0] = mat_inv[0, 0] * x[:, :, 0] + mat_inv[0, 1] * x[:, :, 1] + mat_inv[0, 2] * x[:, :, 2] - offset[0]
#     y[:, :, 1] = mat_inv[1, 0] * x[:, :, 0] + mat_inv[1, 1] * x[:, :, 1] + mat_inv[1, 2] * x[:, :, 2] - offset[1]
#     y[:, :, 2] = mat_inv[2, 0] * x[:, :, 0] + mat_inv[2, 1] * x[:, :, 1] + mat_inv[2, 2] * x[:, :, 2] - offset[2]
#     return y * 255.0

#   path = '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/HCI_new/herbs.h5'

#   with h5py.File(path, 'r') as hf:
#     Hr = np.expand_dims(np.array(hf.get('Hr_SAI_y')), axis=2)
#     Lr = np.expand_dims(np.array(hf.get('Lr_SAI_y')), axis=2)
#     Sr_uv = np.array(hf.get('Sr_SAI_cbcr'))  # [2, 3000, 2000]

#     Hr = np.concatenate([Hr, np.transpose(Sr_uv, (1, 2, 0))], axis=2)

#     rgb = ycbcr2rgb(Hr * 255.0)
#     rgb = np.transpose(rgb, (1, 0, 2))

#     imageio.imwrite('herbs_vanilla.png', np.clip(rgb, 0, 255.0).astype('uint8'))


#     rgb_t = torch.from_numpy(rgb)
#     rgb_t = einops.rearrange(rgb_t, 'h w c -> 1 c h w')
#     u, v = 5, 5

#     rgb_t_1 = einops.rearrange(rgb_t, 'n c (u h) (v w) -> n c (u v) (h w)', u=u, v=v)
#     imageio.imwrite('herbs_uvhw.png', rgb_t_1[0].permute(1, 2, 0).clamp(0, 255.0).numpy().astype('uint8'))

#     rgb_t_1 = einops.rearrange(rgb_t, 'n c (u h) (v w) -> n c (u v) (h w)', u=u, v=v)
#     imageio.imwrite('herbs_uvhw.png', rgb_t_1[0].permute(1, 2, 0).clamp(0, 255.0).numpy().astype('uint8'))
