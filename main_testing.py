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
import pathlib
import argparse

from tqdm import tqdm
from scipy.io import loadmat

# Own codes
from libs.models import ResNetModified
from libs.utilities import load_model, makedir

#%%

def img2rois(img_ld, lambda_e):
    
    h, w = img_ld.shape
    
    # How many Rois fit in the image?
    n_h = h % 64
    n_w = w % 64
    
    if n_h == 0:
        h_pad = h
    else:
        h_pad = (h//64 + 1) * 64
        
    if n_w == 0:
        w_pad = w
    else:
        w_pad = (w//64 + 1) * 64
    
    # Calculate how much padding is necessary and sum 64 for the frontiers
    padding = (((h_pad - h)//2 + 64, (h_pad - h)//2 + 64),
               ((w_pad - w)//2 + 64, (w_pad - w)//2 + 64))
    
    # Pad the image
    img_ld_pad = np.pad(img_ld, padding, mode='reflect')
    lmbd_e_pad = np.pad(lambda_e, padding, mode='reflect')
            
    n_h = h_pad // 64 
    n_w = w_pad // 64 
    
    # Allocate memory to speed up the for loop
    rois = np.empty((n_h*n_w, 2, 192, 192), dtype='float32')

    nRoi = 0
    # Get the ROIs
    for i in range(n_h):
        for j in range(n_w):
            rois[nRoi, 0, :, :] = img_ld_pad[i*64: (i+3)*64, j*64:(j+3)*64]
            rois[nRoi, 1, :, :] = lmbd_e_pad[i*64: (i+3)*64, j*64:(j+3)*64]
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
            rst_img[(i+1)*64:(i+2)*64, (j+1)*64:(j+2)*64] = rst_rois[nRoi,0,64:128,64:128]
            nRoi += 1
        
    org_h, org_w = original_shape
    pad_h, pad_w = padded_shape
    
    # How much to crop?
    start_w = (pad_w - org_w) // 2
    start_h = (pad_h - org_h) // 2
    
    # Crop image
    rst_img = rst_img[start_h:start_h+org_h, start_w:start_w+org_w]
        
    return rst_img
         
def model_forward(model, img_ld, lambda_e, red_factor, batch_size):
    
    # Change model to eval
    model.eval() 

    # Extract ROIs
    rois, padded_shape = img2rois(img_ld, lambda_e)
    
    # Allocate memory to speed up the for loop 
    rst_rois = np.empty_like(rois)
    
    for x in range(0,rois.shape[0],batch_size):
        
        # Get the batch and send to GPU
        batch = torch.from_numpy(rois[x:x+batch_size]).to(device)
        
        # Forward through the model
        with torch.no_grad():
            batch = model(batch)
        
        # Get from GPU
        rst_rois[x:x+batch_size] = batch.to('cpu').numpy()

    # Contruct the image
    rst_img = rois2img(rst_rois, img_ld.shape, padded_shape)
    
    return rst_img

def test(model, path_data, path2write, red_factor, mAsLowDose, batch_size):
    
    path_data_ld = path_data + '31_' + str(mAsLowDose)
    
    file_names = list(pathlib.Path(path_data_ld).glob('**/*.dcm'))

    for file_name in tqdm(file_names):
        
        file_name = str(file_name) 
        
        proj_num = int(file_name.split('/')[-1].split('.')[0][4:]) 
                
        # Read dicom image
        dcmH = pydicom.dcmread(file_name)

        # Read dicom image pixels
        img_ld = dcmH.pixel_array.astype('float32')
        
        lambda_e = lambda_e_nproj[:,-img_ld.shape[1]:,proj_num]       
     
        # Forward through model
        rst_img = model_forward(model, img_ld, lambda_e, red_factor, batch_size)
        
        folder_name = path2write + 'DBT_DL_' + model_type + '_' + file_name.split('/')[-2] 
        file2write_name = 'DL_' + file_name.split('/') [-1]
        
        # Create output dir (if needed)
        makedir(folder_name)
        
        # Copy the restored data to the original dicom header
        dcmH.PixelData = np.uint16(rst_img)
        
        # Write dicom
        pydicom.dcmwrite(os.path.join(folder_name,file2write_name),
                         dcmH, 
                         write_like_original=True) 

    return

#%%

if __name__ == '__main__':
    
    ap = argparse.ArgumentParser(description='Restore low-dose mamography')
    ap.add_argument("--rf", type=int, default=50, required=True, 
                    help="Reduction factor in percentage. (default: 50)")
    ap.add_argument("--rfton", type=int, default=50, required=True, 
                    help="Reduction factor which the model was trained. (default: 50)")
    ap.add_argument("--model", type=str, required=True, 
                    help="Model type")
    
    # sys.argv = sys.argv + ['--rf', '50', '--rfton', '50', '--model', 'VSTasLayer-MNSE_rnw0.33800761'] 
    
    args = vars(ap.parse_args())
        
    model_type = args['model'] 
    
    if model_type == 'Noise2Sim':
        modelSavedNoStandard = True
    else:
        modelSavedNoStandard = False
        
    red_factor = args['rf'] / 100
    red_factor_int = args['rf']
    
        
    # Noise scale factor
    mAsFullDose = 60
    mAsLowDose = int(mAsFullDose * red_factor)
    
    batch_size = 50
        
    path_data = "/media/rodrigo/Data/images/UPenn/Phantom/Anthropomorphic/DBT/"
    path_models = "final_models/"
    path2write = "/media/rodrigo/Data/images/UPenn/Phantom/Anthropomorphic/DBT/Restorations/31_{}/".format(mAsLowDose)
    
    Parameters_Hol_DBT_R_CC_All = loadmat('/media/rodrigo/Data/Estimativas_Parametros_Ruido/Hologic/DBT/Rodrigo/Parameters_Hol_DBT_R_CC_All.mat')

    tau = Parameters_Hol_DBT_R_CC_All['tau'][0][0]
    lambda_e_nproj = Parameters_Hol_DBT_R_CC_All['lambda']
    sigma_e = Parameters_Hol_DBT_R_CC_All['sigma_E'][0][0]
    
    del Parameters_Hol_DBT_R_CC_All
    
    # Test if there is a GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    makedir(path2write)
    
        
    path_final_model = path_models + "model_ResResNet_DBT_{}_{:d}.pth".format(model_type, args['rfton'])
    
    maxGAT = 100#62
    minGAT = 19.935268#58

    # Create model
    model = ResNetModified(tau, sigma_e, red_factor, maxGAT, minGAT)

    # Load pre-trained model parameters (if exist)
    _ = load_model(model, path_final_model=path_final_model, amItesting=True, modelSavedNoStandard=modelSavedNoStandard)
    
    # Send it to device (GPU if exist)
    model = model.to(device)
    
    # Set it to eval mode
    model.eval()
    
    print("Running test on {}.".format(device))
    
    test(model, path_data, path2write, red_factor, mAsLowDose, batch_size)
    
    
    

    
