#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:28:51 2022

@author: Rodrigo
"""

import os 
import pathlib
import torch
import matplotlib.pyplot as plt

from collections import OrderedDict

def load_model(model, optimizer=None, scheduler=None, path_final_model='', path_pretrained_model='', amItesting=False, modelSavedNoStandard=False):
    """Load pre-trained model, resume training or initialize from scratch."""
    
    epoch = 0
      
    # Resume training
    if os.path.isfile(path_final_model):
          
        checkpoint = torch.load(path_final_model, map_location='cuda:0')
      
        if modelSavedNoStandard:
          # create new OrderedDict that does not contain `module.`
          state_dict = checkpoint['state_dict']
          new_state_dict = OrderedDict()
          for k, v in state_dict.items():
              if 'module' in k:
                  name = k[7:] # remove `module.`
              elif 'base_net' in k:
                  name = k[9:] # remove `module.`
              else:
                  name = k
              new_state_dict[name] = v
          # load params
          model.load_state_dict(new_state_dict)
          
        else:  
          model.load_state_dict(checkpoint['model_state_dict'])
          
        if optimizer != None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler != None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch'] + 1
        
        print('Loading model {} from epoch {}.'.format(path_final_model, epoch-1))
      
    # Loading pre-trained model (Initialize optimizer and scheduler from scratch)
    elif os.path.isfile(path_pretrained_model):
          
      # Load a pre trained network 
      checkpoint = torch.load(path_pretrained_model, map_location='cuda:0')
    
      if modelSavedNoStandard:
        # create new OrderedDict that does not contain `module.`
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' in k:
                name = k[7:] # remove `module.`
            elif 'base_net' in k:
                name = k[9:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        
      else:  
        model.load_state_dict(checkpoint['model_state_dict'])
      
      print('Initializing from scratch \nLoading pre-trained {}.'.format(path_pretrained_model))
      
    # Initializing from scratch
    else:
      if amItesting:
        raise ValueError('I couldnt find any model')
      print('I couldnt find any model, I am just initializing from scratch.')
      
    return epoch


def image_grid(ld_img, hd_img, rt_img):
    """Return a 1x3 grid of the images as a matplotlib figure."""
    
    # Get from GPU
    ld_img = ld_img.to('cpu')
    hd_img = hd_img.to('cpu')
    rt_img = rt_img.to('cpu').detach()
    
    # Create a figure to contain the plot.
    figure = plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(torch.squeeze(ld_img),'gray')
    plt.title("Low dose"); plt.grid(False)
    
    plt.subplot(1,3,2)
    plt.imshow(torch.squeeze(hd_img),'gray')
    plt.title("Full dose"); plt.grid(False)
    
    plt.subplot(1,3,3)
    plt.imshow(torch.squeeze(rt_img),'gray')
    plt.title("Restored dose"); plt.grid(False)
      
    return figure

def makedir(path2create):
    """Create directory if it does not exists."""
 
    error = 1
    
    if not os.path.exists(path2create):
        os.makedirs(path2create)
        error = 0
    
    return error

def removedir(directory):
    """Link: https://stackoverflow.com/a/49782093/8682939"""
    directory = pathlib.Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            removedir(item)
        else:
            item.unlink()
    directory.rmdir()

