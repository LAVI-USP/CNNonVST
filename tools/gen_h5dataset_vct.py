#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:28:51 2022

@author: Rodrigo
"""

import numpy as np
import matplotlib.pyplot as plt
import pydicom as dicom
import h5py
import sys
import pywt

from pathlib import Path
from scipy.io import loadmat
from PIL import Image
from tqdm import tqdm

sys.path.insert(1, '../')

from libs.utilities import makedir, removedir

#%%

def VST(img, lambda_e):
    
    # Subtract offset and divide it by the gain of the quantum noise
    img_norm = (img - tau)/lambda_e

    # Apply GAT (Generalized Anscombe VST)    
    img = 2 * np.sqrt(img_norm + 3./8. + sigma_e**2)
    
    return img

def sigmaWaveletAWGN(img):
    # 2D Discrete Wavelet Transform.
    coeffs2 = pywt.dwt2(img, 'db1')

    # Approximation, horizontal detail, vertical detail and diagonal
    # detail coefficients respectively.
    cA, (cH, cV, cD) = coeffs2

    sigma_hat = np.median(np.abs(cD)) / 0.674489750196082

    return sigma_hat

def calcSigmaGAT(img):
    # Image Size
    M, N = img.shape

    # Region to crop
    ii = np.arange((M / 2) - 150 - 1, (M / 2) + 150, 1, dtype='int')
    jj = np.arange(200 - 1, 400, 1, dtype='int')
    jj = N - jj

    sigmaGAT = sigmaWaveletAWGN(img[np.ix_(ii, jj)])

    return sigmaGAT

def get_img_bounds(img):
    '''Get image bounds of the segmented breast from the GT'''
    
    # For GT
    mask = img < 4000
    
    # w
    mask_w = np.sum(mask, 1) > 0
    res = np.where(mask_w == True)
    w_min, w_max = res[0][0], res[0][-1]
    
    # h
    mask_h = np.sum(mask, 0) > 0
    res = np.where(mask_h == True)
    h_min, h_max = res[0][0], res[0][-1]
        
    return w_min, h_min, w_max, h_max


def extract_rois(img_gt, img_fd, img_ld, lambda_e_nproj, rlz, img_id_offset):
    '''Extract low-dose and full-dose rois'''
    
    # Check if images are the same size
    assert img_ld.shape == img_fd.shape == img_gt.shape == lambda_e_nproj.shape, "image sizes differ"
    
    global trow_away, img_id, sigmas, min_global_img, max_global_img
    
    # Get image bounds of the segmented breast from the GT
    w_min, h_min, w_max, h_max = get_img_bounds(img_gt)
    
    # Crop all images
    img_gt = img_gt[w_min:w_max, h_min:h_max]
    img_fd = img_fd[w_min:w_max, h_min:h_max]
    img_ld = img_ld[w_min:w_max, h_min:h_max]
    lambda_e_nproj = lambda_e_nproj[w_min:w_max, h_min:h_max]

    img_fd_vst = VST(img_fd, lambda_e_nproj)
    img_ld_vst = VST(img_ld, lambda_e_nproj)
    sigmas.append(calcSigmaGAT(img_fd_vst))
    sigmas.append(calcSigmaGAT(img_fd_vst))

    # Get updated image shape
    w, h = img_gt.shape
    
    roi_count = 0
    
    # Non-overlaping roi extraction
    for i in range(0, w-64, 64):
        for j in range(0, h-64, 64):
            
            # Extract roi
            patch = (img_ld[i:i+64, j:j+64], 
                     img_fd[i:i+64, j:j+64],
                     img_gt[i:i+64, j:j+64],
                     lambda_e_nproj[i:i+64, j:j+64])
            
            # Am I geting at least one pixel from the breast?
            if np.sum(patch[2]>4000) < (0.7*(64*64)):

                local_min = img_ld_vst[i:i+64, j:j+64].min()
                local_max = img_ld_vst[i:i+64, j:j+64].max()

                if local_min < min_global_img:
                    min_global_img = local_min

                if local_max > max_global_img:
                    max_global_img = local_max

                im_l = Image.fromarray(patch[0])
                im_l.save(gen_path + 'low_rlz{}_id{:06d}.tif'.format(rlz,roi_count+img_id_offset))
                
                im_h = Image.fromarray(patch[1])
                im_h.save(gen_path + 'hgh_rlz{}_id{:06d}.tif'.format(rlz,roi_count+img_id_offset))
                
                if rlz == 1:
                    im_gt = Image.fromarray(patch[2])
                    im_gt.save(gen_path + 'grt_id{:06d}.tif'.format(roi_count+img_id_offset))
                    
                    im_lmb = Image.fromarray(patch[3])
                    im_lmb.save(gen_path + 'lmb_id{:06d}.tif'.format(roi_count+img_id_offset))
                
                roi_count += 1              

    return roi_count


def process_each_folder(folder_name):
    '''Process DBT folder to extract low-dose and full-dose rois'''
    
    global img_id
    
    noisy_path = path2read + 'noisy/' + folder_name.split('/')[-1]
    
    randProjInd = np.random.randint(low=0,high=14,size=14)
        
    rois = []
    
    for proj in randProjInd:
        gt_file_name = noisy_path + '-CM/_{}.dcm'.format(proj)
        
        img_gt = dicom.read_file(gt_file_name).pixel_array
  
        for rlz in range(1, 6):
            
            fd_file_name = noisy_path + '-{}mAs-rlz{}/_{}.dcm'.format(mAsFullDose, rlz, proj)
            ld_file_name = noisy_path + '-{}mAs-rlz{}/_{}.dcm'.format(mAsLowDose, rlz, proj)
        
            img_ld = dicom.read_file(ld_file_name).pixel_array
            img_fd = dicom.read_file(fd_file_name).pixel_array
    
            roi_count = extract_rois(img_gt, img_fd, img_ld, lambda_e_nproj[:, -img_ld.shape[1]:, proj], rlz, img_id)
            
            if rlz == 5:       
                img_id += roi_count
                    
    return rois


#%%

if __name__ == '__main__':
    
    path2read = '/media/rodrigo/Dados_2TB/Imagens/UPenn/Phantom/VCT/VCT_Hologic/'
    path2write = '../data/'
    
    folder_names = [str(item) for item in Path(path2read).glob("*-proj") if Path(item).is_dir()]
    
    redFactor = 50
    
    mAsFullDose = 60
    mAsLowDose = int(mAsFullDose * (redFactor/100))
    
    nROIs_perRlz = 18130
    
    np.random.seed(0)
    
    img_id = 0
    sigmas = []
    
    min_global_img = np.inf
    max_global_img = 0
    
    gen_path = '{}/genROIS_{}/'.format(path2write, redFactor)
    
    makedir(gen_path)
        
    # Create h5 file
    f = h5py.File('{}DBT_VCT_training_{}mAs.h5'.format(path2write, mAsLowDose), 'a')
    
    Parameters_Hol_DBT_L_CC_All = loadmat('/media/rodrigo/Dados_2TB/Estimativas_Parametros_Ruido/Hologic/DBT/Rodrigo/Parameters_Hol_DBT_R_CC_All_VCT.mat')

    tau = Parameters_Hol_DBT_L_CC_All['tau'][0][0]
    lambda_e_nproj = Parameters_Hol_DBT_L_CC_All['lambda']
    sigma_e = Parameters_Hol_DBT_L_CC_All['sigma_E'][0][0]
    
    del Parameters_Hol_DBT_L_CC_All
        
    # Loop on each DBT folder (projections)
    for folder_name in tqdm(folder_names):
        
        # Get low-dose and full-dose rois
        process_each_folder(folder_name)
                
    imgL2save = [] 
    imgH2save = [] 
    imgGT2save = []
    save_loop = 0
    
    for i in range(img_id):
        
        for rlz in range(1,6):
            
            img_l = gen_path + 'low_rlz{}_id{:06d}.tif'.format(rlz,i)
            img_h = gen_path + 'hgh_rlz{}_id{:06d}.tif'.format(rlz,i)
            img_lmb = gen_path + 'lmb_id{:06d}.tif'.format(i)
            im_l = Image.open(img_l)
            im_l = np.asarray(im_l)
            im_h = Image.open(img_h)
            im_h = np.asarray(im_h)
            im_lmb = Image.open(img_lmb)
            im_lmb = np.asarray(im_lmb)
            imgL2save.append(np.concatenate((np.expand_dims(im_l,axis=0),
                                        np.expand_dims(im_lmb,axis=0)),
                                       axis=0))
            imgH2save.append(np.expand_dims(im_h,axis=0))
            img_gt = gen_path + 'grt_id{:06d}.tif'.format(i)
            im_gt = Image.open(img_gt)
            im_gt = np.asarray(im_gt)
            imgGT2save.append(np.expand_dims(im_gt,axis=0))
    
        
        if (i+1) % 10 == 0:
            
            imgL2save = np.stack(imgL2save,axis=0)
            imgH2save = np.stack(imgH2save,axis=0)
            imgGT2save = np.stack(imgGT2save,axis=0)
            
            if save_loop == 0:
                f.create_dataset('data', data=imgL2save, chunks=True, maxshape=(None,2,64,64))
                f.create_dataset('label', data=imgH2save, chunks=True, maxshape=(None,1,64,64)) 
                f.create_dataset('gt', data=imgGT2save, chunks=True, maxshape=(None,1,64,64))
            else:
                f['data'].resize((f['data'].shape[0] + imgL2save.shape[0]), axis=0)
                f['data'][-imgL2save.shape[0]:] = imgL2save
                
                f['label'].resize((f['label'].shape[0] + imgH2save.shape[0]), axis=0)
                f['label'][-imgH2save.shape[0]:] = imgH2save
                
                f['gt'].resize((f['gt'].shape[0] + imgGT2save.shape[0]), axis=0)
                f['gt'][-imgGT2save.shape[0]:] = imgGT2save
                
            print("I am on iteration {} and 'data' chunk has shape:{} and 'label':{}".format(save_loop,f['data'].shape,f['label'].shape))
            
            imgL2save = [] 
            imgH2save = [] 
            imgGT2save = []
            save_loop += 1

    f.close() 
    removedir(gen_path)
