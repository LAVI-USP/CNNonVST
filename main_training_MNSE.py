#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:28:51 2022

@author: Rodrigo
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import sys
import argparse

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.io import loadmat

# Own codes
from libs.models import ResResNet, UNet2, RED, ResNet
from libs.utilities import load_model, image_grid, makedir
from libs.dataset import BreastCancerVCTDataset
from libs.losses import MNSE


def train(model, optimizer, epoch, train_loader, device, summarywriter, rnw):
    # Enable trainning

    loss_min = 100

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

        summarywriter.add_scalar('rnw * rn_loss',
                                 rnw * rn_loss,
                                 epoch * len(train_loader) + step)

        summarywriter.add_scalar('RN',
                                 rn_loss,
                                 epoch * len(train_loader) + step)

        if step % 20 == 0:
            summarywriter.add_figure('Plot/train',
                                     image_grid(data[0, 0, :, :],
                                                target[0, 0, :, :],
                                                rst_data[0, 0, :, :]),
                                     epoch * len(train_loader) + step,
                                     close=True)

        # Save the lowest bias
        if loss < loss_min:
            loss_min = loss

            # Save the model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path_final_model)

    return


# %%

if __name__ == '__main__':

    ap = argparse.ArgumentParser(description='Restore low-dose mamography')

    ap.add_argument("--nep", type=int, default=2, required=True,
                    help="Number of epochs. (default: 50)")
    ap.add_argument("--rnw", type=float, default=0.1, required=True,
                    help="Residual noise weight. (default: 50)")
    ap.add_argument("--model", type=str, default='', required=True,
                    help="Model architecture")

    # sys.argv = sys.argv + ['--rnw', '0.0000', '--model', 'resnet', '--nep', '2']

    args = vars(ap.parse_args())

    seed = 1639
    torch.manual_seed(seed)
    np.random.seed(seed)

    rnw = args['rnw']

    # Noise scale factor
    red_factor = 0.5
    red_factor_self_learning = 50  # red_factor which self learning was trained
    red_factor_int = int(red_factor * 100)

    # Noise scale factor
    mAsFullDose = 60
    mAsLowDose = int(mAsFullDose * red_factor)

    path_data = "data/"

    Parameters_Hol_DBT_R_CC_All = loadmat(path_data + 'Parameters_Hol_DBT_R_CC_All.mat')

    tau = Parameters_Hol_DBT_R_CC_All['tau'][0][0]
    sigma_e = Parameters_Hol_DBT_R_CC_All['sigma_E'][0][0]

    del Parameters_Hol_DBT_R_CC_All

    bond_val_vst = {100: (358.9964, 59.1849),
                    50: (420.777562, 19.935268),  # VST min/max (58.34536070368842, 417.52899640547685)
                    25: (297.289434, 14.042236),
                    15: (234.938067, 7.301423),
                    5: (137.023591, 3.6612093)}

    maxGAT = bond_val_vst[red_factor_self_learning][0]
    minGAT = bond_val_vst[red_factor_self_learning][1]

    # print(minGAT, maxGAT)

    # Create model
    if args['model'] == 'RED':
        model = RED(tau, sigma_e, red_factor, maxGAT, minGAT)
    elif args['model'] == 'UNet2':
        model = UNet2(tau, sigma_e, red_factor, maxGAT, minGAT, residual=True)
    elif args['model'] == 'ResResNet':
        model = ResResNet(tau, sigma_e, red_factor, maxGAT, minGAT)
    else:
        raise ValueError('I couldnt find any model')

    path_models = "final_models/"
    path_logs = "final_logs/{}-rnw{}-r{}-{}".format(model.__class__.__name__,
                                                    rnw,
                                                    red_factor_int,
                                                    time.strftime("%Y-%m-%d-%H%M%S", time.localtime()))

    path_pretrained_model = path_models + "model_{}_DBT_Noise2Sim_{:d}.pth".format(model.__class__.__name__,
                                                                                   red_factor_self_learning)

    LR = 0.0001 / 10
    batch_size = 60
    n_epochs = args['nep']

    dataset_name = '{}DBT_VCT_training_{:d}mAs.h5'.format(path_data, mAsLowDose)

    if batch_size % 5 != 0:
        raise ValueError('Batch size need to be multiple of 5')

    # Tensorboard writer
    summarywriter = SummaryWriter(log_dir=path_logs)

    makedir(path_models)
    makedir(path_logs)

    # Test if there is a GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create the optimizer and the LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 2, 3, 4, 5], gamma=0.5)

    # Send it to device (GPU if exist)
    model = model.to(device)

    # Load pre-trained model parameters (if exist)
    _ = load_model(model,
                   optimizer,
                   None,
                   '',
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
    for epoch in range(n_epochs):
        print("Epoch:[{}] LR:{}".format(epoch, scheduler.get_last_lr()))


        path_final_model = path_models + "model_{}_DBT_VSTasLayer-MNSE_rnw{}_{:d}.pth".format(
            model.__class__.__name__,
            rnw,
            red_factor_int)


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

