#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 10:05:18 2022

@author: rodrigo
"""

import pydicom
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import os
# from scipy.io import loadmat
# import png

path2Read = '/media/rodrigo/Dados_2TB/Imagens/UPenn/Phantom/Anthropomorphic/DBT/'

#%% Image parameters

nRows = 1788    
nCols = 1664 

mAsFullDose = 60
reducFactors = [50]

DL_types = ['ResResnet-VSTasLayer-MNSE_rnw0.01357164',
            'ResResnet-VSTasLayer-MNSE_rnw0.33800761',
            'ResResnet-VSTasLayer-MNSE_rnw0.94957105',
            'ResResnet-Noise2Sim',
            'ResResnet-PL4',
            'ResResnet-L1']

n_all_fullDose = 20
n_rlzs_fullDose = 10
n_rlzs_GT = 10
DL_ind = 0

ii_phantom, jj_phantom  = 1226, 1346  
ii_phantom, jj_phantom  = 385, 1373
ii_phantom, jj_phantom  = 485, 1413

w = 200#43

if not os.path.exists('phantom_ROI'):
    os.mkdir('phantom_ROI')

#%% Read Dicom function

def readDicom(dir2Read, imgSize):
    
    # List dicom files
    dcmFiles = list(pathlib.Path(dir2Read).glob('*.dcm'))
    
    dcmImg = np.empty([imgSize[0],imgSize[1]])
    
    if not dcmFiles:    
        raise ValueError('No DICOM files found in the specified path.')

    for dcm in dcmFiles:
        
        
        ind = int(str(dcm).split('/')[-1].split('_')[-1].split('.')[0])
        
        if ind == 7:
            dcmH = pydicom.dcmread(str(dcm))

            dcmImg[:,:] = dcmH.pixel_array[130:-130,:].astype('float32') #[0:,1128:].astype('float32') 
    
    return dcmImg

def mat2gray(img, rangeVal):
    
    # level_value = 555
    # window_value = 232
    
    # vmin = level_value - (window_value // 2)
    # vmax = level_value + (window_value // 2)
    # print(vmin, vmax)

    # img = np.clip(img, img, vmax)
    
    img = np.clip(img, rangeVal[0], rangeVal[1])
    
    img = (img - img.min()) / ((img.max() - img.min()) / 255.) 
    
    return img    

def save_figure(img, fname):


    print(img.min(),img.max())    

    img = np.uint8(255 - mat2gray(img,(408, 703)))#498, 663)))
    plt.imsave('phantom_ROI/' + fname + '.png', img, cmap=plt.cm.gray)


#%% Read clinical data

# Read all full dose images
fullDose_all = np.empty(shape=(nRows,nCols,n_all_fullDose))

paths = sorted(list(pathlib.Path(path2Read + "31_" + str(mAsFullDose) ).glob('*/')))

if paths == []:
    raise ValueError('No FD results found.') 

print('Reading FD images...')
for idX, path in enumerate(paths):
    fullDose_all[:,:,idX] = readDicom(path,(nRows,nCols))
       
# Generate the GT - fisrt time
groundTruth = np.mean(fullDose_all[:,:,n_rlzs_GT:], axis=-1)
# Generate a boolean mask for the breast
maskBreast = groundTruth < 2500

z=0
save_figure(fullDose_all[ii_phantom:ii_phantom+w,jj_phantom:jj_phantom+w,z], '60_01_Mammo_R_CC')


              

# Read MB, restored results and reuced doses
for reduc in reducFactors:
    
    mAsReducFactors = int((reduc / 100) * mAsFullDose)
    
    # Reduced doses
    paths = list(pathlib.Path(path2Read + "31_" + str(mAsReducFactors) ).glob('*')) 
    if paths == []:
        raise ValueError('No RD results found.')
    paths = [path for path in paths if not 'MB' in str(path) and not 'DL' in str(path)]
    reduDose_rlzs = np.empty(shape=(nRows,nCols))
    for idX, path in enumerate(paths):
        all_rlzs =  readDicom(path,(nRows,nCols))
        reduDose_rlzs = all_rlzs
        reduDose_rlzs = ((reduDose_rlzs - 50) / (reduc / 100)) + 50
        save_figure(reduDose_rlzs[ii_phantom:ii_phantom+w,jj_phantom:jj_phantom+w], '{}_01_Mammo_R_CC'.format(mAsReducFactors))
        
        break
            

    # MB restored doses
    paths = list(pathlib.Path(path2Read + "Restorations/31_" + str(mAsReducFactors) ).glob('MB*')) 
    if paths == []:
        raise ValueError('No MB results found.')
    restDose_MB_rlzs = np.empty(shape=(nRows,nCols))
    for idX, path in enumerate(paths):
        restDose_MB_rlzs =  readDicom(path,(nRows,nCols))      
        save_figure(restDose_MB_rlzs[ii_phantom:ii_phantom+w,jj_phantom:jj_phantom+w], 'MB_{}_01_Mammo_R_CC'.format(mAsReducFactors))
        break


    # Loop through DL methods
    for indDL, DL_type in enumerate(DL_types):
        
            print('Reading and calculating {}({}mAs) images...'.format(DL_type,mAsReducFactors))
            
            # DL restored doses
            paths = list(pathlib.Path(path2Read + "Restorations/31_" + str(mAsReducFactors) ).glob('DBT_DL_' + DL_type + '*')) 
            if paths == []:
                raise ValueError('No DL results found.')
            restDose_DL_rlzs = np.empty(shape=(nRows,nCols))
            for idZ, path in enumerate(paths):
                restDose_DL_rlzs =  readDicom(path,(nRows,nCols))
                save_figure(restDose_DL_rlzs[ii_phantom:ii_phantom+w,jj_phantom:jj_phantom+w], 'DL-{}_{}_01_Mammo_R_CC'.format(DL_type, mAsReducFactors))
                break
                        