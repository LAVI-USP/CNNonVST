#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:28:51 2022

@author: Rodrigo
"""

import matplotlib.pyplot as plt
import torch
import time
import sys
import argparse

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.io import loadmat

# Own codes
from libs.models import ResNetModified
from libs.utilities import load_model, image_grid, makedir
from libs.dataset import BreastCancerVCTDataset
from libs.losses import MNSE

#%%

def train(model, optimizer, epoch, train_loader, device, summarywriter, rnw):

    # Enable trainning
    model.train()

    for step, (data, target, gt) in enumerate(tqdm(train_loader)):

        data = data.to(device)
        target = target.to(device)
        gt = gt.to(device)

        # Zero all grads            
        optimizer.zero_grad()

        # Pass data through model
        rst_data = model(data)

        # Calc loss
        rst_b2, rst_rn = MNSE(rst_data, gt)
        tgt_b2, tgt_rn = MNSE(target, gt)
        
        b2_loss = rst_b2
        rn_loss = torch.abs(rst_rn - tgt_rn)
        
        loss = b2_loss + rnw * rn_loss

        # Calculate all grads
        loss.backward()
        
        # Update weights and biases based on the calc grads 
        optimizer.step()

        # Write Loss to tensorboard
        summarywriter.add_scalar('Loss/train', 
                                 loss.item(), 
                                 epoch * len(train_loader) + step)
        
        summarywriter.add_scalar('Bias', 
                                 b2_loss, 
                                 epoch * len(train_loader) + step)
        
        summarywriter.add_scalar('wRN', 
                                 rnw * rn_loss, 
                                 epoch * len(train_loader) + step)
        
        if step % 20 == 0:
            summarywriter.add_figure('Plot/train', 
                                     image_grid(data[0,0,:,:], 
                                                target[0,0,:,:], 
                                                rst_data[0,0,:,:]),
                                     epoch * len(train_loader) + step,
                                     close=True)

#%%

if __name__ == '__main__':
    
    ap = argparse.ArgumentParser(description='Restore low-dose mamography')
    ap.add_argument("--rf", type=int, default=50, required=True, 
                    help="Reduction factor in percentage. (default: 50)")
    ap.add_argument("--rnw", type=float, default=0.1, required=True, 
                    help="Residual noise weight. (default: 50)")
    ap.add_argument("--rfton", type=int, default=50, required=True, 
                    help="Reduction factor which the model was trained. (default: 50)")
    
    # sys.argv = sys.argv + ['--rf', '50', '--rnw', '0.01357164', '--rfton', '100'] 
    
    args = vars(ap.parse_args())
    
    rnw = args['rnw']
    
    model_type = 'MNSE'
    
    # Noise scale factor
    red_factor = args['rf'] / 100
    red_factor_int = int(red_factor*100)
    
    # Noise scale factor
    mAsFullDose = 60
    mAsLowDose = int(mAsFullDose * red_factor)
    
    path_data = "data/"
    path_models = "final_models/"
    path_logs = "final_logs/{}-rnw{}-r{}".format(time.strftime("%Y-%m-%d-%H%M%S", time.localtime()), rnw, red_factor_int)
    
    path_final_model = path_models + "model_ResResNet_DBT_VSTasLayer-{}_rnw{}_{:d}.pth".format(model_type,rnw,red_factor_int)
    path_pretrained_model = path_models + "model_ResResNet_DBT_Noise2Sim_{:d}.pth".format(args['rfton']) 
    
    Parameters_Hol_DBT_R_CC_All = loadmat('/media/rodrigo/Dados_2TB/Estimativas_Parametros_Ruido/Hologic/DBT/Rodrigo/Parameters_Hol_DBT_R_CC_All.mat')

    tau = Parameters_Hol_DBT_R_CC_All['tau'][0][0]
    sigma_e = Parameters_Hol_DBT_R_CC_All['sigma_E'][0][0]
    
    del Parameters_Hol_DBT_R_CC_All
    
    LR = 0.0001 / 10.
    batch_size = 120
    n_epochs = 2
    
    bond_val_vst = {100:(358.9964, 59.1849),# 100:(591.989278, 29.463522), #
                    50:(420.777562, 19.935268),#50:(418.270143, 20.751217),
                    25:(297.289434, 14.042236),
                    15:(234.938067, 7.301423),
                    5:(137.023591, 3.6612093)}
    
    maxGAT = bond_val_vst[args['rfton']][0]
    minGAT = bond_val_vst[args['rfton']][1]
    
    print(minGAT, maxGAT)
    
    dataset_name = '{}DBT_VCT_training_{:d}mAs.h5'.format(path_data, mAsLowDose)
    
    
    if batch_size % 5 != 0:
        raise ValueError('Batch size need to be multiple of 5')
    
    
    # Tensorboard writer
    summarywriter = SummaryWriter(log_dir=path_logs)
    
    makedir(path_models)
    makedir(path_logs)
    
    # Test if there is a GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = ResNetModified(tau, sigma_e, red_factor, maxGAT, minGAT)
    
    # Create the optimizer and the LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50], gamma=0.5)
    
    # Send it to device (GPU if exist)
    model = model.to(device)
    
    # Load pre-trained model parameters (if exist)
    start_epoch = load_model(model, 
                             optimizer,
                             None,
                             path_final_model, 
                             path_pretrained_model,
                             modelSavedNoStandard=True)
    
    # Create dataset helper
    train_set = BreastCancerVCTDataset(dataset_name, red_factor, tau)
    
    # Create dataset loader (NOTE: shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_set,
                                              batch_size=batch_size, 
                                              shuffle=False,
                                              pin_memory=True)
        
    # Loop on epochs
    for epoch in range(start_epoch, n_epochs):
        
      print("Epoch:[{}] LR:{}".format(epoch, scheduler.get_last_lr()))
    
      # Train the model for 1 epoch
      train(model, 
            optimizer, 
            epoch, 
            train_loader, 
            device, 
            summarywriter,
            rnw) 
    
      # Update LR
      scheduler.step()
    
      # Save the model
      torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, path_final_model)