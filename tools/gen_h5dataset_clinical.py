#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:28:51 2022

@author: Rodrigo
"""

import numpy as np
import pydicom as dicom
import h5py
import sys

from scipy.ndimage import median_filter, uniform_filter1d
from skimage.filters import threshold_otsu
from pathlib import Path
from scipy.io import loadmat
from PIL import Image

sys.path.insert(1, '../')

from libs.utilities import makedir, removedir

#%%

def VST(img, lambda_e):
    
    # Subtract offset and divide it by the gain of the quantum noise
    img_norm = (img - tau)/lambda_e

    # Apply GAT (Generalized Anscombe VST)    
    img = 2 * np.sqrt(img_norm + 3./8. + sigma_e**2)
    
    return img

def get_img_bounds(img):
    '''Get image bounds of the segmented breast from the GT'''
    
    # for normal dose
    vertProj = np.sum(img,axis=0)
    
    # Smooth the signal and take the first derivative
    firstDv = np.gradient(uniform_filter1d(median_filter(vertProj,size=250), size=50))
    
    # Smooth the signal and take the second derivative
    secondDv = np.gradient(uniform_filter1d(firstDv, size=50))
    
    # Takes its max second derivative
    indX = np.argmin(secondDv) 
    
    # w
    w_min, w_max = 50, img.shape[0]-50
    
    # h
    h_min, h_max = indX, vertProj.shape[0]
    
    thresh = threshold_otsu(img[w_min:w_max, h_min:h_max])
        
    return w_min, h_min, w_max, h_max, thresh


def extract_rois(img_fd_vst, img_ld_vst, img_fd):
    '''Extract low-dose and full-dose rois'''
    
    # Check if images are the same size
    assert img_fd_vst.shape == img_ld_vst.shape, "image sizes differ"
    
    global trow_away, img_id
    
    # Get image bounds of the segmented breast from the GT
    w_min, h_min, w_max, h_max, thresh = get_img_bounds(img_fd)
    
    # Crop all images
    img_fd = img_fd[w_min:w_max, h_min:h_max]
    img_ld_vst = img_ld_vst[w_min:w_max, h_min:h_max]
    img_fd_vst = img_fd_vst[w_min:w_max, h_min:h_max]
    
    # Get updated image shape
    w, h = img_fd.shape
        
    # Non-overlaping roi extraction
    for i in range(0, w-64, 64):
        for j in range(0, h-64, 64):
            
            # Extract roi
            roi_tuple = (img_ld_vst[i:i+64, j:j+64], img_fd_vst[i:i+64, j:j+64])
            
            # Am I geting at least one pixel from the breast?
            if np.sum(img_fd[i:i+64, j:j+64] > thresh) != 64*64:
                
                im_l = Image.fromarray(roi_tuple[0])
                im_l.save(gen_path + 'low_' + "{:06d}".format(img_id) + '.tif')
                
                img_id += 1
                
            else:
                trow_away += 1                



def process_each_folder(folder_name, redFactor, num_proj=15):
    '''Process DBT folder to extract low-dose and full-dose rois'''   
        
    rois = []
    
    # Loop on each projection
    for proj in range(num_proj):
          
        # Full-dose image
        fd_file_name = folder_name + "/{}.dcm".format(proj)

        # Low-dose image
        ld_file_name = folder_name + "/{}_L{}.dcm".format(proj, redFactor)
    
        img_fd = dicom.read_file(fd_file_name).pixel_array
        img_ld = dicom.read_file(ld_file_name).pixel_array
        
        img_fd[img_fd < tau] = tau
        img_ld[img_ld < tau] = tau
        
        lambda_e = lambda_e_nproj[:,-img_fd.shape[1]:,proj]
        
        img_fd_vst = VST(img_fd, lambda_e)
        img_ld_vst = VST(img_ld, lambda_e)
        
        # Check sigma on GAT
        # print(calcSigmaGAT(img_fd))
    
        extract_rois(img_fd_vst, img_ld_vst, img_fd)
                    
    return 

#%%

if __name__ == '__main__':
    
    path2read = '/media/rodrigo/Dados_2TB/Imagens/USP/Inrad/Inrad_Raw'
    path2write = '../data/'
    
    folder_names = [str(item) for item in Path(path2read).glob("*/*") if Path(item).is_dir()]
    
    folder_names = []
    
    redFactor = 50
    
    mAsFullDose = 60
    mAsLowDose = int(mAsFullDose * (redFactor/100))
    
    nROIs_total = 256000
    
    np.random.seed(0)
    
    trow_away = 0
    flag_final = 0
    nROIs = 0
    
    min_global_img = np.inf
    max_global_img = 0
    
    gen_path = '{}/genROIS_{}/'.format(path2write, redFactor)
    
    makedir(gen_path)
    
    # Create h5 file
    f = h5py.File('{}DBT_VST_training_{}mAs.h5'.format(path2write, mAsLowDose), 'a')
    
    Parameters_Hol_DBT_R_CC_All = loadmat('/media/rodrigo/Dados_2TB/Estimativas_Parametros_Ruido/Hologic/DBT/Rodrigo/Parameters_Hol_DBT_R_CC_All.mat')

    tau = Parameters_Hol_DBT_R_CC_All['tau'][0][0]
    lambda_e_nproj = Parameters_Hol_DBT_R_CC_All['lambda']
    sigma_e = Parameters_Hol_DBT_R_CC_All['sigma_E'][0][0]
    
    img_id = 0
    
    # Loop on each DBT folder (projections)
    for idX, folder_name in enumerate(folder_names):
        
        # Get low-dose and full-dose rois
        process_each_folder(folder_name, redFactor)        
                
    rand_index = np.random.permutation(img_id)
        
    for i in rand_index[256000:]:
        file_to_rem = Path(gen_path + 'low_' + "{:06d}".format(i) + '.tif')
        file_to_rem.unlink()                   
