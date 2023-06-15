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

from tqdm import tqdm
from scipy.io import loadmat

# Own codes
from libs.models import ResResNet, UNet2, RED, ResNet
from libs.utilities import load_model, makedir


def VST(img, lambda_e):
    # Subtract offset and divide it by the gain of the quantum noise
    img_norm = (img - tau) / lambda_e

    # Apply GAT (Generalized Anscombe VST)
    img = 2 * np.sqrt(img_norm + 3. / 8. + sigma_e ** 2)

    return img


def img2rois(img_ld, lambda_e):
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
    lmbd_e_pad = np.pad(lambda_e, padding, mode='reflect')

    n_h = h_pad // 64
    n_w = w_pad // 64

    # Allocate memory to speed up the for loop
    rois = np.empty((n_h * n_w, 2, 192, 192), dtype='float32')

    nRoi = 0
    # Get the ROIs
    for i in range(n_h):
        for j in range(n_w):
            rois[nRoi, 0, :, :] = img_ld_pad[i * 64: (i + 3) * 64, j * 64:(j + 3) * 64]
            rois[nRoi, 1, :, :] = lmbd_e_pad[i * 64: (i + 3) * 64, j * 64:(j + 3) * 64]
            nRoi += 1

    return rois, img_ld_pad.shape


def rois2img(rst_rois, original_shape, padded_shape):
    rst_img = np.empty((padded_shape))

    n_h = (padded_shape[0] // 64) - 2
    n_w = (padded_shape[1] // 64) - 2

    nRoi = 0
    # Reconstruct image format
    for i in range(n_h):
        for j in range(n_w):
            rst_img[(i + 1) * 64:(i + 2) * 64, (j + 1) * 64:(j + 2) * 64] = rst_rois[nRoi, 0, 64:128, 64:128]
            nRoi += 1

    org_h, org_w = original_shape
    pad_h, pad_w = padded_shape

    # How much to crop?
    start_w = (pad_w - org_w) // 2
    start_h = (pad_h - org_h) // 2

    # Crop image
    rst_img = rst_img[start_h:start_h + org_h, start_w:start_w + org_w]

    return rst_img


def model_forward(model, img_ld, lambda_e, batch_size):
    # Change model to eval
    model.eval()

    # Extract ROIs
    rois, padded_shape = img2rois(img_ld, lambda_e)

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

        proj_num = int(file_name.split('/')[-1].split('.')[0][4:])

        # Read dicom image
        dcmH = pydicom.dcmread(file_name)

        # Read dicom image pixels
        img_ld = dcmH.pixel_array.astype('float32')

        rst_img = img_ld.copy()

        lambda_e = lambda_e_nproj[:, -img_ld.shape[1]:, proj_num]

        # img_ld_vst = VST(img_ld, lambda_e)
        #
        # mask = img_ld < 993 # 1157:
        #
        # local_min = img_ld_vst[mask].min()
        # local_max = img_ld_vst[mask].max()
        #
        # if local_min < min_global_img:
        #     min_global_img = local_min
        #
        # if local_max > max_global_img:
        #     max_global_img = local_max

        start = time.time()

        # Forward through model
        rst_img[:, 1156:] = model_forward(model, img_ld[:, 1156:], lambda_e[:, 1156:], batch_size)

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


# %%

if __name__ == '__main__':

    ap = argparse.ArgumentParser(description='Restore low-dose mamography')

    ap.add_argument("--rnw", type=float, default=0.1, required=True,
                    help="Residual noise weight. (default: 50)")
    ap.add_argument("--model", type=str, default='', required=True,
                    help="Model architecture")
    ap.add_argument("--fmw", type=str, required=True,
                    help="Framework")

    # sys.argv = sys.argv + ['--rnw', '0.0', '--model', 'ResResNet', '--nep', '2', '--fmw', 'Noise2Sim']#, 'Noise2Sim']

    args = vars(ap.parse_args())

    model_type = args['fmw']
    rnw = args['rnw']
    batch_size = 50

    min_global_img = np.inf
    max_global_img = 0

    # Noise scale factor
    red_factor = 0.5
    red_factor_self_learning = 50  # red_factor which self learning was trained
    red_factor_int = int(red_factor * 100)

    mAsFullDose = 60
    mAsLowDose = int(mAsFullDose * red_factor)

    path_data = "/home/laviusp/Documents/Rodrigo_Vimieiro/phantom/"
    # path_data = '/media/rodrigo/Dados_2TB/Imagens/UPenn/Phantom/Anthropomorphic/DBT/'
    path_models = "final_models/"
    path2write = path_data + "Restorations/31_{}/".format(mAsLowDose)

    Parameters_Hol_DBT_R_CC_All = loadmat('data/Parameters_Hol_DBT_R_CC_All.mat')

    tau = Parameters_Hol_DBT_R_CC_All['tau'][0][0]
    lambda_e_nproj = Parameters_Hol_DBT_R_CC_All['lambda']
    sigma_e = Parameters_Hol_DBT_R_CC_All['sigma_E'][0][0]

    del Parameters_Hol_DBT_R_CC_All

    # Test if there is a GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    makedir(path2write)

    # 181.51 / 58.0 valores em VST do phantom (dentro da mama)
    # 230 / 58.0 valores em VST do phantom (total fora os pixels saturados)
    # 420.777562 / 19.935268 valores em VST clinicas
    maxGAT = 420.777562
    minGAT = 19.935268

    # Create model
    if args['model'] == 'RED':
        model = RED(tau, sigma_e, red_factor, maxGAT, minGAT)
    elif args['model'] == 'UNet2':
        model = UNet2(tau, sigma_e, red_factor, maxGAT, minGAT, residual=True)
    elif args['model'] == 'ResResNet':
        model = ResResNet(tau, sigma_e, red_factor, maxGAT, minGAT)
    else:
        raise ValueError('Unknown model')

    if model_type == 'Noise2Sim':
        modelSavedNoStandard = True
        model_description = "{}_DBT_Noise2Sim_{:d}".format(model.__class__.__name__,
                                                           red_factor_self_learning)
        path_final_model = path_models + "model_{}.pth".format(model_description)

    else:
        modelSavedNoStandard = False
        model_description = "{}_DBT_VSTasLayer-MNSE_rnw{}_{:d}".format(model.__class__.__name__,
                                                                       rnw,
                                                                       red_factor_int)
        path_final_model = path_models + "model_{}.pth".format(model_description)

    # Load pre-trained model parameters (if exist)
    _ = load_model(model, path_final_model=path_final_model, amItesting=True, modelSavedNoStandard=modelSavedNoStandard)

    # Send it to device (GPU if exist)
    model = model.to(device)

    # Set it to eval mode
    model.eval()

    print("Running test on {}. of 31_{}mAs images".format(device, mAsLowDose))

    test(model, path_data, path2write, mAsLowDose, batch_size)
