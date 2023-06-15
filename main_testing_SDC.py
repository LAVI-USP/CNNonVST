#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:28:51 2022

@author: Rodrigo
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import pydicom
import os
import time
import pathlib
import argparse
import sys

from tqdm import tqdm

# Own codes
from libs.models import ResResNet_SDC, UNet2_SDC, RED_SDC
from libs.utilities import load_model, makedir
from main_testing import rois2img
from libs.dataset import scale, de_scale

# %%

def img2rois(img_ld):
    h, w = img_ld.shape

    # How many Rois fit in the image?
    n_h = h % 64
    n_w = w % 64

    if n_h == 0:
        h_pad = h
    else:
        h_pad = (h // 64 + 1) * 64

    if n_w == 0:
        w_pad = w
    else:
        w_pad = (w // 64 + 1) * 64

    # Calculate how much padding is necessary and sum 64 for the frontiers
    padding = (((h_pad - h) // 2 + 64, (h_pad - h) // 2 + 64),
               ((w_pad - w) // 2 + 64, (w_pad - w) // 2 + 64))

    # Pad the image
    img_ld_pad = np.pad(img_ld, padding, mode='reflect')

    n_h = h_pad // 64
    n_w = w_pad // 64

    # Allocate memory to speed up the for loop
    rois = np.empty((n_h * n_w, 1, 192, 192), dtype='float32')

    nRoi = 0
    # Get the ROIs
    for i in range(n_h):
        for j in range(n_w):
            rois[nRoi, 0, :, :] = img_ld_pad[i * 64: (i + 3) * 64, j * 64:(j + 3) * 64]
            nRoi += 1

    return rois, img_ld_pad.shape

def model_forward(model, img_ld, batch_size):
    global min_global_img, max_global_img
    # Change model to eval
    model.eval()

    # Extract ROIs
    rois, padded_shape = img2rois(img_ld)

    # local_min = rois.min()
    # local_max = rois.max()
    #
    # if local_min < min_global_img:
    #     min_global_img = local_min
    #
    # if local_max > max_global_img:
    #     max_global_img = local_max

    rois = scale(rois, vmin, vmax, red_factor=red_factor)

    # Allocate memory to speed up the for loop
    rst_rois = np.empty_like(rois)

    for x in range(0, rois.shape[0], batch_size):
        # Get the batch and send to GPU
        batch = torch.from_numpy(rois[x:x + batch_size]).to(device)

        # Forward through the model
        with torch.no_grad():
            batch = model(batch)

        # Get from GPU
        rst_rois[x:x + batch_size] = batch.to('cpu').numpy()

    rst_rois = de_scale(rst_rois, vmin, vmax)

    # Construct the image
    rst_img = rois2img(rst_rois, img_ld.shape, padded_shape)

    return rst_img


def test(model, path_data, path2write, mAsLowDose, batch_size):
    global min_global_img, max_global_img

    path_data_ld = path_data + '31_' + str(mAsLowDose)

    file_names = list(pathlib.Path(path_data_ld).glob('**/*.dcm'))

    elapsed_times = []
    for file_name in tqdm(file_names):
        file_name = str(file_name)

        # Read dicom image
        dcmH = pydicom.dcmread(file_name)

        # Read dicom image pixels
        img_ld = dcmH.pixel_array.astype('float32')

        rst_img = img_ld.copy()

        # mask = img_ld < 1000
        #
        # local_min = img_ld[mask].min()
        # local_max = img_ld[mask].max()
        #
        # if local_min < min_global_img:
        #     min_global_img = local_min
        #
        # if local_max > max_global_img:
        #     max_global_img = local_max

        start = time.time()

        # Forward through model
        rst_img[:, 1156:] = model_forward(model, img_ld[:, 1156:], batch_size)

        end = time.time()
        elapsed_times.append(end - start)

        folder_name = path2write + model_description + '_' + file_name.split('/')[-2]
        file2write_name = 'DL_' + file_name.split('/')[-1]

        # Create output dir (if needed)
        makedir(folder_name)

        # Copy the restored data to the original dicom header
        dcmH.PixelData = np.uint16(rst_img)

        # Write dicom
        pydicom.dcmwrite(os.path.join(folder_name, file2write_name),
                         dcmH,
                         write_like_original=True)

    print(np.mean(elapsed_times))

    return

if __name__ == '__main__':

    ap = argparse.ArgumentParser(description='Restore low-dose mamography')

    ap.add_argument("--model", type=str, default='', required=True,
                    help="Model architecture")
    ap.add_argument("--fmw", type=str, required=False,
                    help="Loss")

    # sys.argv = sys.argv + ['--model', 'UNet2', '--fmw', 'PL4']

    args = vars(ap.parse_args())

    model_loss = args['fmw']

    batch_size = 50

    min_global_img = np.inf
    max_global_img = 0

    # 999 / 211 valores do phantom (dentro da mama)
    # 2039 / 211 valores do phantom (ROIs)
    # 13560 / 48 valores Clinicas (ROIs)
    vmin = 48.
    vmax = 13560.

    # Noise scale factor
    red_factor = 0.5
    red_factor_int = int(red_factor * 100)

    mAsFullDose = 60
    mAsLowDose = int(mAsFullDose * red_factor)

    # path_data = "/home/laviusp/Documents/Rodrigo_Vimieiro/phantom/"
    path_data = '/media/rodrigo/Dados_2TB/Imagens/UPenn/Phantom/Anthropomorphic/DBT/'
    path_models = "final_models/"
    path2write = path_data + "Restorations/31_{}/".format(mAsLowDose)

    # Test if there is a GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    makedir(path2write)

    # Create model
    if args['model'] == 'RED':
        model = RED_SDC()
    elif args['model'] == 'UNet2':
        model = UNet2_SDC(residual=True)
    elif args['model'] == 'ResResNet':
        model = ResResNet_SDC()
    else:
        raise ValueError('Unknown model')

    modelSavedNoStandard = False
    model_description = "{}_DBT_{}_{:d}".format(model.__class__.__name__.replace("_SDC", ""),
                                                model_loss,
                                                red_factor_int)
    path_final_model = path_models + "model_{}.pth".format(model_description)

    # Load pre-trained model parameters (if exist)
    _ = load_model(model, path_final_model=path_final_model, amItesting=True)

    # Send it to device (GPU if exist)
    model = model.to(device)

    # Set it to eval mode
    model.eval()

    print("Running test on {}. of 31_{}mAs images".format(device, mAsLowDose))

    test(model, path_data, path2write, mAsLowDose, batch_size)