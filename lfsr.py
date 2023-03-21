# Copyright 2022 The KaiJIN Authors. All Rights Reserved.
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
"""Light-Field Super Resolution: based on BasicLSFR

    https://github.com/ZhengyuLiang24/BasicLFSR

  Train:
    Input is y-channel from ycbcr

  Input Mode:
    1) single image [H, W, 3]
    2) batch image [U, V, H, W, 3]

"""
import glob
import os
import random
import tqdm
import copy
import argparse
import functools
import einops

import cv2
import numpy as np
from skimage import metrics
import h5py
import imageio

import mat73
import scipy.io

import torch
from torch.nn import functional as F
from torch import nn
import einops

import imresize
import models


PROTOCALS = {
    'LFSR.ORG': {
        'train': [
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/EPFL/training/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/HCI_new/training/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/HCI_old/training/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/INRIA_Lytro/training/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/Stanford_Gantry/training/',
        ],
        'val': [
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/EPFL/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/HCI_new/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/HCI_old/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/INRIA_Lytro/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/Stanford_Gantry/',
        ]
    },
    'LFSR.ALL3': {
        'train': [
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/EPFL/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/HCI_new/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/HCI_old/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/INRIA_Lytro/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/Stanford_Gantry/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_13579/EPFL/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_13579/HCI_new/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_13579/HCI_old/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_13579/INRIA_Lytro/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_13579/Stanford_Gantry/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_24568/EPFL/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_24568/HCI_new/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_24568/HCI_old/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_24568/INRIA_Lytro/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_24568/Stanford_Gantry/',
        ],
        'val': [
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/EPFL/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/HCI_new/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/HCI_old/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/INRIA_Lytro/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/Stanford_Gantry/',
        ]
    },
    'LFSR.ALL2': {
        'train': [
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/EPFL/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/HCI_new/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/HCI_old/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/INRIA_Lytro/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/Stanford_Gantry/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_13579/EPFL/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_13579/HCI_new/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_13579/HCI_old/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_13579/INRIA_Lytro/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_13579/Stanford_Gantry/',
        ],
        'val': [
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/EPFL/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/HCI_new/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/HCI_old/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/INRIA_Lytro/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/Stanford_Gantry/',
        ]
    },
    'LFSR.ALL': {
        'train': [
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/EPFL/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/HCI_new/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/HCI_old/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/INRIA_Lytro/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32_matlab/Stanford_Gantry/',
        ],
        'val': [
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/EPFL/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/HCI_new/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/HCI_old/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/INRIA_Lytro/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x_matlab/Stanford_Gantry/',
        ]
    },
    'LFSR.EPFL': {
        'train': ['/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32/EPFL/'],
        'val': ['/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x/EPFL/']
    },
    'LFSR.HCInew': {
        'train': ['/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32/HCI_new/'],
        'val': ['/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x/HCI_new/']
    },
    'LFSR.HCIold': {
        'train': ['/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32/HCI_old/'],
        'val': ['/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x/HCI_old/']
    },
    'LFSR.INRIA': {
        'train': ['/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32/INRIA_Lytro/'],
        'val': ['/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x/INRIA_Lytro/']
    },
    'LFSR.STFgantry': {
        'train': ['/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_training/SR_5x5_4x_128_32/Stanford_Gantry/'],
        'val': ['/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_test/SR_5x5_4x/Stanford_Gantry/']
    },
    'LFSR.NTIRE.VAL': {
        'test': [
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_inference/SR_5x5_4x/NTIRE_Val_Real/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_inference/SR_5x5_4x/NTIRE_Val_Synth/'
        ]
    },
    'LFSR.NTIRE.TEST': {
        'test': [
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_inference/SR_5x5_4x/NTIRE_Test_Real/',
            '/cephFS/video_lab/datasets/super_resolution/LFSR_NTIRE_2023/data_for_inference/SR_5x5_4x/NTIRE_Test_Synth/'
        ]
    }
}


def restoration_augment(img, mode):
  if mode == 0:
    return img
  elif mode == 1:
    return np.flipud(np.rot90(img))
  elif mode == 2:
    return np.flipud(img)
  elif mode == 3:
    return np.rot90(img, k=3)
  elif mode == 4:
    return np.flipud(np.rot90(img, k=2))
  elif mode == 5:
    return np.rot90(img)
  elif mode == 6:
    return np.rot90(img, k=2)
  elif mode == 7:
    return np.flipud(np.rot90(img, k=3))


def rgb2ycbcr(x):
  """rgb to ycbcr

  Args:
      x (np.ndarray): [H, W, 3] (0, 255) range

  Returns:
      y (np.ndarray): [H, W, 3] (16, 235) range
  """
  x = x / 255.0
  y = np.zeros(x.shape, dtype='double')
  y[:, :, 0] = 65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] + 24.966 * x[:, :, 2] + 16.0
  y[:, :, 1] = -37.797 * x[:, :, 0] - 74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
  y[:, :, 2] = 112.000 * x[:, :, 0] - 93.786 * x[:, :, 1] - 18.214 * x[:, :, 2] + 128.0
  # y = y / 255.0
  return y


def ycbcr2rgb(x):
  """ycbcr to rgb

  Args:
      y (np.ndarray): [H, W, 3] (16, 235) range

  Returns:
      x (np.ndarray): [H, W, 3] (0, 255) range
  """
  x = x / 255.0
  mat = np.array(
      [[65.481, 128.553, 24.966],
       [-37.797, -74.203, 112.0],
       [112.0, -93.786, -18.214]])
  mat_inv = np.linalg.inv(mat)
  offset = np.matmul(mat_inv, np.array([16, 128, 128]))
  mat_inv = mat_inv * 255

  y = np.zeros(x.shape, dtype='double')
  y[:, :, 0] = mat_inv[0, 0] * x[:, :, 0] + mat_inv[0, 1] * x[:, :, 1] + mat_inv[0, 2] * x[:, :, 2] - offset[0]
  y[:, :, 1] = mat_inv[1, 0] * x[:, :, 0] + mat_inv[1, 1] * x[:, :, 1] + mat_inv[1, 2] * x[:, :, 2] - offset[1]
  y[:, :, 2] = mat_inv[2, 0] * x[:, :, 0] + mat_inv[2, 1] * x[:, :, 1] + mat_inv[2, 2] * x[:, :, 2] - offset[2]
  return y * 255.0


def to_tensor(inputs: np.ndarray, scale=None, mean=None, std=None, **kwargs) -> np.ndarray:
  # mean = torch.tensor(mean) if mean is not None else None
  # std = torch.tensor(std) if std is not None else None

  if inputs.ndim == 3:
    m = torch.from_numpy(np.ascontiguousarray(inputs.transpose((2, 0, 1))))
  elif inputs.ndim == 2:
    m = torch.from_numpy(inputs)[None]
  elif inputs.ndim == 4:
    m = torch.from_numpy(np.ascontiguousarray(inputs.transpose((0, 3, 1, 2))))
  else:
    raise NotImplementedError(inputs.ndim)

  m = m.type(torch.FloatTensor)
  if scale is not None:
    m = m.float().div(scale)
  if mean is not None:
    m.sub_(torch.tensor(mean)[:, None, None])
  if std is not None:
    m.div_(torch.tensor(std)[:, None, None])
  return m


def transform_test_lfsr(lr_img, uv_img, path, scale=4):
  """input ycbcr [0, 1.0] float
  """
  lr_img = to_tensor(lr_img, scale=1.0)
  return lr_img, uv_img, path


def transform_val_lfsr(lr_img, hr_img, scale=4):
  """input ycbcr [0, 1.0] float
  """
  lr_img = to_tensor(lr_img, scale=1.0)
  hr_img = to_tensor(hr_img, scale=1.0)
  return lr_img, hr_img


def transform_train_lfsr(lr_img, hr_img):
  """input ycbcr [0, 1.0] float
  """
  # randomly crop the HR patch
  H, W, C = lr_img.shape

  # augmentation - flip and/or rotate
  mode = random.randint(0, 7)
  lr_img = restoration_augment(lr_img, mode=mode)
  hr_img = restoration_augment(hr_img, mode=mode)

  # convert to tensor
  lr_img = to_tensor(lr_img, scale=1.0)
  hr_img = to_tensor(hr_img, scale=1.0)
  return lr_img, hr_img


class ModelWrapper(nn.Module):
  """Video noise reduction wrapper
  """

  def __init__(self, net_g: nn.Module, net_d: nn.Module = None, net_e: nn.Module = None):
    super(ModelWrapper, self).__init__()
    self.netG = net_g
    self.netD = net_d
    self.netE = net_e

  def forward(self, x, gt=None):
    if gt is not None:
      return self.netG(x, gt=gt)
    return self.netG(x)


class LFSRDataset(torch.utils.data.Dataset):

  """Light-Field Dataset

  Data Format:
    1) h5 format:
    2) npy format:
    3) png/bmp format:

  """

  def __init__(self, phase, paths, transform, transpose=True, **kwargs):
    """LFSR dataset with h5 file

    Args:
        phase (_type_): _description_
        paths (_type_): _description_
        transform (_type_): _description_
        mode (str, optional): _description_. Defaults to 'single'.
    """
    self.targets = []
    self.phase = phase
    self.angular = 5
    self.scale = 4
    self.patch_size = 32
    self.transpose = transpose

    if self.phase == 'train':
      # if use .mat data, should repeat to 300
      repeat = 1
    else:
      repeat = 1

    # collect files
    files = []
    for path in paths:
      files.extend(sorted(glob.glob(f'{path}/**/*.h5', recursive=True)))
      files.extend(sorted(glob.glob(f'{path}/**/*.mat', recursive=True)))

    # preprocess
    self.cache = {}
    for path in tqdm.tqdm(files):
      if path.endswith('.mat'):
        self.cache[path] = self.load_mat(path)

    self.targets = files * repeat
    self.transform = transform
    print(f'total load num of image: {len(self.targets)}.')

    self.count = 0

  def load_mat(self, path):
    try:
      LF = np.array(scipy.io.loadmat(path)['LF'])
    except:
      LF = np.array(mat73.loadmat(path)['LF'])
    return LF

  def parse_mat(self, path):
    if self.phase != 'train':
      raise NotImplementedError(self.phase)

    LF = np.copy(self.cache[path])

    U, V, H, W, _ = LF.shape

    # augmentation for random select 5x5 patches
    if U == V == 9:
      # select 5 row
      select_row = [4, ]
      select_row.extend(random.sample([0, 1, 2, 3], 2))
      select_row.extend(random.sample([5, 6, 7, 8], 2))
      select_row = sorted(select_row)
      # select 5 col
      select_col = [4, ]
      select_col.extend(random.sample([0, 1, 2, 3], 2))
      select_col.extend(random.sample([5, 6, 7, 8], 2))
      select_col = sorted(select_col)
      LF = LF[select_row][:, select_col]

    # random crop a patch
    U, V, H, W, _ = LF.shape
    patch_size = self.patch_size * self.scale
    rnd_h = random.randint(0, max(0, H - patch_size))
    rnd_w = random.randint(0, max(0, W - patch_size))
    LF = LF[:, :, rnd_h: rnd_h + patch_size, rnd_w: rnd_w + patch_size, 0:3]

    # convert to sai views
    LF = np.transpose(LF, (0, 2, 1, 3, 4)).reshape([U * patch_size, V*patch_size, 3])

    # convert to yuv and select y only
    Sr = rgb2ycbcr(LF * 255.0) / 255.0
    Sr_uv = Sr[..., 1:3]
    Hr = Sr[..., 0:1]
    Lr = imresize.imresize(Hr, scalar_scale=1.0/self.scale)

    return Lr, Hr, Sr_uv

  def parse_h5(self, path):
    with h5py.File(path, 'r') as hf:
      Hr = np.expand_dims(np.array(hf.get('Hr_SAI_y')), axis=2)
      Lr = np.expand_dims(np.array(hf.get('Lr_SAI_y')), axis=2)
      Sr_uv = np.array(hf.get('Sr_SAI_cbcr'))  # [2, 3000, 2000]

      if self.transpose:
        Hr = np.transpose(Hr, (1, 0, 2))
        Lr = np.transpose(Lr, (1, 0, 2))
        if Sr_uv.ndim > 0:
          Sr_uv = np.transpose(Sr_uv, (0, 2, 1))

      return Lr, Hr, Sr_uv

  def __len__(self):
    return len(self.targets)

  def __getitem__(self, idx):
    """fetch elements
    """
    filepath = self.targets[idx]

    if filepath.endswith('.h5'):
      lr, hr, uv = self.parse_h5(filepath)
    elif filepath.endswith('.mat'):
      lr, hr, uv = self.parse_mat(filepath)
    else:
      raise NotImplementedError(filepath)

    if self.phase != 'test':
      return self.transform(lr, hr)
    else:
      return self.transform(lr, uv, filepath)


class LFSR():
  """Light-Field Super-Resolution
  """

  def __init__(self, config):
    self.Config = config

    # overwrite when dist
    self.Master = True

    # scalar
    self.Epoch = 0
    self.Step = 0

    # models
    self.Model = self.build_model()
    self.ModelEMA = None

    # optim
    self.Optim = self.build_optim(self.Model)
    self.Loss = None

    # load model and possible optimizer
    if self.Config.model_path is not None:
      self.load(self.Model, self.Config.model_source, self.Config.model_path)

    # extend to distributed
    if self.Config.task == 'train':
      self.Model = torch.nn.DataParallel(self.Model)

  def dump(self):
    """dump current checkpoint
    """
    cfg = self.Config
    path = f'{cfg.root}/model.epoch-{self.Epoch}.step-{self.Step}.pth'
    torch.save({
        'state_dict': self.Model.state_dict(),
        'global_step': self.Step,
        'global_epoch': self.Epoch,
        'optimizer': self.Optim.state_dict(),
    }, path)
    print(f'Model has saved in {path}')

    if cfg.ema_decay > 0:
      path = f'{cfg.root}/model.epoch-{self.Epoch}.step-{self.Step}.ema.pth'
      torch.save({
          'state_dict': self.ModelEMA.state_dict(),
          'global_step': self.Step,
          'global_epoch': self.Epoch,
          'optimizer': self.Optim.state_dict(),
      }, path)
      print(f'Model has saved in {path}')

  def load(self, model: nn.Module, model_source: str, model_path: str):
    """Loading model
    """
    print('Loading model source: {}'.format(model_source))
    ckpt = torch.load(model_path)
    state_dict = {}
    for k, v in ckpt['state_dict'].items():
      state_dict[k.replace('module.', '')] = v

    if model_source == 'wzq':
      model.netG.load_state_dict(state_dict)
    else:
      model.load_state_dict(state_dict)

  def build_model(self):
    """build models
    """
    cfg = self.Config
    device = self.Config.device
    angular, scale = cfg.angular, cfg.scale

    # DistgEPIT-series
    if cfg.model == 'DistgEPIT_deeper':
      netG = models.DistgEPIT_deeper(angRes_in=angular, scale_factor=scale)
    elif cfg.model == 'DistgEPIT_wider':
      netG = models.DistgEPIT_wider(angRes_in=angular, scale_factor=scale)
    elif cfg.model == 'DistgEPIT_v6':
      netG = models.DistgEPIT_v6(angRes_in=angular, scale_factor=scale)
    elif cfg.model == 'DistgSSR':
      netG = models.DistgSSR(angRes_in=angular, scale_factor=scale)
    else:
      raise NotImplementedError(cfg.model)

    model = ModelWrapper(net_g=netG)
    model.to(device)

    return model

  def build_optim(self, model: nn.Module):
    """build optimizer
    """
    cfg = self.Config
    if cfg.task == 'train':
      optim = torch.optim.Adam(model.parameters(), lr=cfg.train_lr, weight_decay=0.0)
    else:
      optim = None
    return optim

  def build_dataset(self, phase):
    """build dataset
    """
    cfg = self.Config
    paths = PROTOCALS[cfg.dataset]

    if phase == 'train':
      dataset = LFSRDataset(phase=phase, paths=paths['train'], transform=transform_train_lfsr)
    elif phase == 'val':
      dataset = LFSRDataset(phase=phase, paths=paths['val'], transform=transform_val_lfsr)
    elif phase == 'test':
      dataset = LFSRDataset(phase=phase, paths=paths['test'], transform=transform_test_lfsr)
    else:
      raise NotImplementedError(phase)

    return dataset

  def build_dataloader(self, dataset, phase):
    """build dataloader
    """
    if phase == 'train':
      return torch.utils.data.DataLoader(
          dataset=dataset,
          batch_size=self.Config.train_batchsize,
          shuffle=True,
          num_workers=1,
          pin_memory=False,
          drop_last=True)
    else:
      return torch.utils.data.DataLoader(
          dataset=dataset,
          batch_size=1,
          shuffle=False,
          num_workers=1,
          pin_memory=False,
          drop_last=False)

  def inference(self, lr_images):
    """general inference pipeline
    """
    cfg = self.Config
    device = cfg.device

    if cfg.ema_decay > 0:
      model = self.ModelEMA
    else:
      model = self.Model

    hr_preds = model(lr_images)

    return hr_preds

  def optimize(self, lr_images, hr_images):
    """optimize model
    """
    hr_preds = self.Model(lr_images)
    losses = {'loss_l1': self.loss_l1(hr_preds, hr_images)}
    return losses

  def train(self):
    """training
    """
    cfg = self.Config
    device = self.Config.device
    init_step = self.Step

    # build train dataset
    train_set = self.build_dataset('train')
    train_loader = self.build_dataloader(train_set, 'train')
    total_step = len(train_loader) * cfg.train_epoch
    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(self.Optim, step_size=80, gamma=0.5)  # S4
    # losses
    self.loss_l1 = nn.L1Loss()

    # EMA
    if cfg.ema_decay > 0:
      self.ModelEMA = copy.deepcopy(self.Model.eval())
      print(f'using EMA technique to training with {cfg.ema_decay}')

    # training
    loss_stat = 0.0
    while self.Epoch < cfg.train_epoch:
      self.Epoch += 1
      self.Model.train()

      # training a epoch
      for lr_images, hr_images in train_loader:

        self.Step += 1
        lr_images = lr_images.float().to(device)
        hr_images = hr_images.float().to(device)

        # inference
        losses = self.optimize(lr_images, hr_images)

        # accumulate
        loss = sum(loss for loss in losses.values())
        self.Optim.zero_grad()
        loss.backward()
        self.Optim.step()

        # ema
        if cfg.ema_decay > 0:
          params = dict(self.Model.named_parameters())
          ema = dict(self.ModelEMA.named_parameters())
          for i, k in enumerate(params):
            ema[k].data.mul_(cfg.ema_decay).add_(params[k].data, alpha=1 - cfg.ema_decay)

        # iter
        loss_stat += sum(loss for loss in losses.values())
        if self.Step % cfg.log == 0:
          print(self.Step, loss_stat.item() / cfg.log)
          loss_stat = 0

      if self.Epoch % cfg.log_save == 0 and self.Master:
        self.dump()

      if self.Epoch % cfg.log_val == 0 and self.Master:
        self.val_all()

      # learning rate scheduler
      lr_scheduler.step()

  def tta(self, lr_images):
    """TTA based on _restoration_augment_tensor
    """
    hr_preds = []
    hr_preds.append(self.inference(lr_images))
    hr_preds.append(self.inference(lr_images.rot90(1, [2, 3]).flip([2])).flip([2]).rot90(3, [2, 3]))
    hr_preds.append(self.inference(lr_images.flip([2])).flip([2]))
    hr_preds.append(self.inference(lr_images.rot90(3, [2, 3])).rot90(1, [2, 3]))
    hr_preds.append(self.inference(lr_images.rot90(2, [2, 3]).flip([2])).flip([2]).rot90(2, [2, 3]))
    hr_preds.append(self.inference(lr_images.rot90(1, [2, 3])).rot90(3, [2, 3]))
    hr_preds.append(self.inference(lr_images.rot90(2, [2, 3])).rot90(2, [2, 3]))
    hr_preds.append(self.inference(lr_images.rot90(3, [2, 3]).flip([2])).flip([2]).rot90(1, [2, 3]))
    return torch.stack(hr_preds, dim=0).mean(dim=0)

  @torch.no_grad()
  def compute_psnr_ssim(self, sr, gt):
    """compute psnr and ssim index in terms of matlab version

    Args:
        sr (torch.Tensor): [N, 1, H, W] in [0, 1] float
        gt (torch.Tensor): [N, 1, H, W] in [0, 1] float

    Returns:
        dict: {'psnr': v, 'ssim': v}
    """
    cfg = self.Config
    assert sr.size(0) == gt.size(0) == 1, "Current only support batchsize to 1."

    # following basic lfsr mode: using float32 (0, 1) to compute
    gt = gt[0][0].cpu().numpy()
    sr = sr[0][0].cpu().numpy()
    psnr = metrics.peak_signal_noise_ratio(gt, sr)
    ssim = metrics.structural_similarity(gt, sr, gaussian_weights=True)

    return {'psnr': psnr, 'ssim': ssim}

  @torch.no_grad()
  def val(self, loader=None, **kwargs):
    """inference with model (noise image, gt image)
    """
    cfg = self.Config
    device = self.Config.device

    # reset
    self.Model.eval()

    # dataset
    if loader is None:
      dataset = self.build_dataset('val')
      loader = self.build_dataloader(dataset, 'val')

    # create folder for every epoch
    root = os.makedirs(f'{cfg.root}/val/epoch_{self.Epoch}_step_{self.Step}/', exist_ok=True)
    reports = {'psnr': [], 'ssim': []}

    # start
    step, total = 0, int(len(loader))
    for lr_images, hr_images in loader:
      # count
      step += 1

      lr_images = lr_images.float().to(device)
      hr_images = hr_images.float().to(device)

      # inference
      hr_preds = self.inference_no_pad(lr_images)

      # evaluate
      res = self.compute_psnr_ssim(hr_preds, hr_images)
      print('{}/{}, psnr: {}, ssim: {}'.format(step, total, res['psnr'], res['ssim']))
      reports['psnr'].append(res['psnr'])
      reports['ssim'].append(res['ssim'])

    reports['psnr'] = np.mean(reports['psnr'])
    reports['ssim'] = np.mean(reports['ssim'])
    return reports

  @torch.no_grad()
  def val_all(self):
    """validation for all lfsr dataset
    """
    total_psnr = []
    total_ssim = []
    for dataset in ['LFSR.EPFL', 'LFSR.HCInew', 'LFSR.HCIold', 'LFSR.INRIA', 'LFSR.STFgantry']:
      self.Config.dataset = dataset
      reports = self.val()
      total_psnr.append(float(reports['psnr']))
      total_ssim.append(float(reports['ssim']))

    print('Epoch:{}, Iter:{}, mean_psnr: {:.4f}, mean_ssim: {:.4f}'.format(
        self.Epoch, self.Step, np.mean(total_psnr), np.mean(total_ssim)))

  @torch.no_grad()
  def test(self, **kwargs):
    """inference with model (noise image, gt image)
    """
    cfg = self.Config
    device = cfg.device
    angular = cfg.angular

    # reset
    self.Model.eval()

    # dataset
    dataset = self.build_dataset('test')
    loader = self.build_dataloader(dataset, 'test')

    # create folder for every epoch
    root = os.makedirs(f'{cfg.root}/test/epoch_{self.Epoch}_step_{self.Step}/', exist_ok=True)
    cache = {
        'path': cfg.model_path,
        'name': cfg.name,
        'pred': [],
        'path': [],
    }

    # start
    step, total = 0, int(len(loader))
    for y_lr, uv_sr, filepath in tqdm.tqdm(loader):

      # count
      step += 1
      y_lr = y_lr.float().to(device)
      uv_sr = uv_sr.float().to(device)

      # inference
      y_pred = self.inference_no_pad(y_lr)
      yuv_sr = torch.cat([y_pred, uv_sr], dim=1)  # [1, 3, h, w] in (0, 1)

      # cache
      cache['pred'].append(yuv_sr)
      cache['path'].append(filepath)

      # construct dst folder
      folder, name = filepath[0].split('/')[-2:]
      folder = 'Real' if 'Real' in folder else 'Synth'
      dstdir = f'{root}/{folder}/{name[:-3]}/'
      os.makedirs(dstdir, exist_ok=True)

      # save sub-image to folder
      yuv_sr = einops.rearrange(yuv_sr, '1 c (u h) (v w) -> u v h w c', u=angular, v=angular)
      yuv_sr = yuv_sr.cpu().mul(255.0).numpy()
      for i in range(angular):
        for j in range(angular):
          out = np.clip(ycbcr2rgb(yuv_sr[i, j]), 0, 255).astype('uint8')
          imageio.imwrite(f'{dstdir}/View_{i}_{j}.bmp', out)

    # save
    torch.save(cache, os.path.join(root, 'test.pth'))

    # save
    pwd = os.getcwd()
    os.system(f'cd {root} && zip -r submission.zip Real Synth')
    os.system(f'cd {pwd}')

  @torch.no_grad()
  def lf_divide_no_pad(self, lr_img):
    """lr_img [n, c, (u h), (v w)]
    """
    cfg = self.Config
    angular = cfg.angular
    stride = cfg.test_stride
    patch_size = cfg.test_patch

    sub_lf, numU = [], 0
    lr_img = einops.rearrange(lr_img, 'n c (u h) (v w) -> n c u v h w', u=angular, v=angular)
    [_, _, u0, v0, h0, w0] = lr_img.size()

    sub_lf = []
    for y in range(0, h0, stride):
      for x in range(0, w0, stride):
        if y + patch_size > h0 and x + patch_size <= w0:
          sub_lf.append(lr_img[..., h0 - patch_size:, x: x + patch_size])
        elif y + patch_size <= h0 and x + patch_size > w0:
          sub_lf.append(lr_img[..., y: y + patch_size, w0 - patch_size:])
        elif y + patch_size > h0 and x + patch_size > w0:
          sub_lf.append(lr_img[..., h0 - patch_size:, w0 - patch_size:])
        else:
          sub_lf.append(lr_img[..., y: y + patch_size, x: x + patch_size])

    return torch.concat(sub_lf, dim=0)

  @torch.no_grad()
  def lf_integrate_no_pad(self, lf_divided, lr_sai_h, lr_sai_w):
    """
    """
    cfg = self.Config
    angular = cfg.angular
    stride = cfg.test_stride * cfg.scale
    patch_size = cfg.test_patch * cfg.scale
    scale = cfg.scale

    # each SAI size
    h0 = lr_sai_h // angular
    w0 = lr_sai_w // angular
    h1 = h0 * scale
    w1 = w0 * scale

    # rearrange to SAI views
    lf_divided = einops.rearrange(lf_divided, 'n c (u h) (v w) -> n c u v h w', u=angular, v=angular)
    _, c, u, v, h, w = lf_divided.size()

    # allocate space
    out = torch.zeros(c, u, v, h1, w1).to(lf_divided.device)
    mask = torch.zeros(c, u, v, h1, w1).to(lf_divided.device)

    idx = 0
    for y in range(0, h1, stride):
      for x in range(0, w1, stride):
        if y + patch_size > h1 and x + patch_size <= w1:
          out[..., h1 - patch_size:, x: x + patch_size] += lf_divided[idx]
          mask[..., h1 - patch_size:, x: x + patch_size] += 1
        elif y + patch_size <= h1 and x + patch_size > w1:
          out[..., y: y + patch_size, w1 - patch_size:] += lf_divided[idx]
          mask[..., y: y + patch_size, w1 - patch_size:] += 1
        elif y + patch_size > h1 and x + patch_size > w1:
          out[..., h1 - patch_size:, w1 - patch_size:] += lf_divided[idx]
          mask[..., h1 - patch_size:, w1 - patch_size:] += 1
        else:
          out[..., y: y + patch_size, x: x + patch_size] += lf_divided[idx]
          mask[..., y: y + patch_size, x: x + patch_size] += 1
        idx += 1

    return out / mask

  @torch.no_grad()
  def inference_no_pad(self, lr_images):
    """inference based on wzq
    """
    cfg = self.Config
    device = cfg.device
    scale = cfg.scale
    angular = cfg.angular
    stride = cfg.test_stride
    patch_size = cfg.test_patch
    h_in, w_in = lr_images.shape[-2:]

    # [70, 1, 5, 5, 32, 32]
    sub_lf = self.lf_divide_no_pad(lr_images)

    # [70, 1, 32 * 5, 32 * 5]
    sub_lf = einops.rearrange(sub_lf, 'n c u v h w -> n c (u h) (v w)', u=angular, v=angular)
    b, c, h, w = sub_lf.shape

    # store output
    sub_lf_out = torch.zeros(b, c, h * scale, w * scale).to(device)

    # overlapping each patches
    for i in range(b):
      sub_inp = sub_lf[i].reshape(1, c, h, w)
      if cfg.tta:
        sub_out = self.tta(sub_inp)
      else:
        sub_out = self.inference(sub_inp)
      sub_lf_out[i] = sub_out

    # integrate into one image
    sub_lf_out = self.lf_integrate_no_pad(sub_lf_out, h_in, w_in)
    hr_preds = einops.rearrange(sub_lf_out, 'c u v h w -> 1 c (u h) (v w)')

    return hr_preds

  def __call__(self):
    """prepare basic
    """
    cfg = self.Config

    if cfg.task == 'train':
      self.train()

    elif cfg.task == 'val':
      self.val()

    elif cfg.task == 'val_all':
      self.val_all()

    elif cfg.task == 'test':
      self.test()

    else:
      raise NotImplementedError(cfg.task)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # ---------------------------------------------
  #  USED BY CONTEXT
  # ---------------------------------------------
  parser.add_argument('--name', type=str, default='tw')
  parser.add_argument('--root', type=str, default=None, help="None for creating, otherwise specific root.")
  parser.add_argument('--device', type=str, default='cuda:0')
  parser.add_argument('--output_dir', type=str, default='_outputs', help="default output folder.")

  # ---------------------------------------------
  #  USED BY COMMON
  # ---------------------------------------------
  parser.add_argument('--task', type=str, default=None, choices=['train', 'val', 'val_all', 'test'])
  parser.add_argument('--dataset', type=str, default=None)

  # ---------------------------------------------
  #  USED BY LOGGER
  # ---------------------------------------------
  parser.add_argument('--log', type=int, default=10, help="display interval step.")
  parser.add_argument('--log-val', type=int, default=1, help="running validation in terms of step.")
  parser.add_argument('--log-save', type=int, default=1, help="saveing checkpoint with interval.")

  # ---------------------------------------------
  #  USED BY MODEL-SPECIFIC
  # ---------------------------------------------
  parser.add_argument('--model', type=str, default=None, help="")
  parser.add_argument('--model-path', type=str, default=None, help="loadding pretrain/last-checkpoint model.")
  parser.add_argument('--model-source', type=str, default=None)

  # ---------------------------------------------
  #  USED BY TRAIN-SPECIFIC
  # ---------------------------------------------
  parser.add_argument('--train-batchsize', type=int, default=32, help="total batch size across devices.")
  parser.add_argument('--train-epoch', type=int, default=100, help="total training epochs.")
  parser.add_argument('--train-lr', type=float, default=2e-4, help="training learning rate.")
  parser.add_argument('--input-colorspace', type=str, default='Y', choices=['Y', 'YUV', 'RGB'])

  # ---------------------------------------------
  #  USED BY INPUT-SPECIFIC
  # ---------------------------------------------
  parser.add_argument('--scale', type=int, default=4, help="upsample scale.")
  parser.add_argument('--ema-decay', type=float, default=0.0, help="using EMA techniques.")
  parser.add_argument('--angular', type=int, default=5, choices=[5, 9])

  # only for test tta
  parser.add_argument('--tta', action='store_true', help="test time augmentation.")
  parser.add_argument('--test-patch', type=int, default=32)
  parser.add_argument('--test-stride', type=int, default=16)

  # ---------------------------------------------
  #  USED BY VIZ-SPECIFIC
  # ---------------------------------------------
  parser.add_argument('--viz-input', type=str, default=None, help='input path could be a folder/filepath.')
  parser.add_argument('--viz-output', type=str, help='output path should be a folder path.')

  config, _ = parser.parse_known_args()
  config.root = "%s/%s" % (config.output_dir, config.name)
  os.makedirs(config.root, exist_ok=True)

  # run
  LFSR(config)()
