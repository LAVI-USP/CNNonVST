#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:28:51 2022

@author: Rodrigo
"""

import torch
import h5py


def de_scale(data, vmin, vmax):
        
    data = data * vmax 
    
    data = data + vmin
        
    return data

def scale(data, vmin, vmax, red_factor=1.):
    
    data -= vmin
    data /= red_factor
    data += vmin
    data /= vmax
    
    data[data > 1.0] = 1.0
    data[data < 0.0] = 0.0
          
    return data


class BreastCancerVCTDataset(torch.utils.data.Dataset):
  """Breast Cancer VCT dataset."""
  def __init__(self, h5_file_name, red_factor, tau):
    """
    Args:
      h5_file_name (string): Path to the h5 file.
      transform (callable, optional): Optional transform to be applied
          on a sample.
    """
    self.h5_file_name = h5_file_name
    self.red_factor = red_factor
    
    self.tau = tau

    self.h5_file = h5py.File(self.h5_file_name, 'r')

    self.data = self.h5_file['data']
    self.target = self.h5_file['label']
    self.gt = self.h5_file['gt']

  def __len__(self):
    return self.data.shape[0]


  def __getitem__(self, idx):

    if torch.is_tensor(idx):
      idx = idx.tolist()

    data = self.data[idx,:,:,:]
    target = self.target[idx,:,:,:]
    gt = self.gt[idx,:,:,:]

    # To torch tensor
    data = torch.from_numpy(data.astype(float)).type(torch.FloatTensor)
    target = torch.from_numpy(target.astype(float)).type(torch.FloatTensor)
    gt = torch.from_numpy(gt.astype(float)).type(torch.FloatTensor)
    
    # Check if we have data less than Tau. Also greater than 0 to avoid changing lambda
    data[(data > 1) & (data <self.tau )] = self.tau 

    return data, target, gt

