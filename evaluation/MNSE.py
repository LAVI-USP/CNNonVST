#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:28:51 2022

@author: Rodrigo
"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pydicom
import pandas as pd
import argparse
import sys

# from scipy.io import loadmat
# from scipy.ndimage import binary_erosion
# from skimage.morphology import disk

# sys.path.insert(1,'../')

import pyeval

#%% Read Dicom function

def readDicom(dir2Read, imgSize):
    
    # List dicom files
    dcmFiles = list(pathlib.Path(dir2Read).glob('*.dcm'))
    
    dcmImg = np.empty([imgSize[0],imgSize[1],len(dcmFiles)])
    
    if not dcmFiles:    
        raise ValueError('No DICOM files found in the specified path.')

    for dcm in dcmFiles:
        
        dcmH = pydicom.dcmread(str(dcm))
        
        ind = int(str(dcm).split('/')[-1].split('_')[-1].split('.')[0])
        
        dcmImg[:,:,ind] = dcmH.pixel_array[130:-130,50:-50].astype('float32')
    
    return dcmImg

#%%

if __name__ == '__main__':
    
    ap = argparse.ArgumentParser(description='Restore low-dose mamography')
    ap.add_argument("--rf", type=int, default=50, required=True, 
                    help="Reduction factor in percentage. (default: 50)")
    ap.add_argument("--model", type=str, required=True, 
                    help="Model type")
    
    # sys.argv = sys.argv + ['--rf', '50', '--model', 'ResResNet_DBT_VSTasLayer-MNSE_rnw0.42105263']#'ResNet_DBT_Noise2Sim']#VSTasLayer-MNSE_ep_1rnw0.0']
    
    args = vars(ap.parse_args())
        
    model_type = args['model'] 
    
    #%% General parameters
    
    path2Read = '/home/laviusp/Documents/Rodrigo_Vimieiro/phantom/'
    # path2Read = '/media/rodrigo/Dados_2TB/Imagens/UPenn/Phantom/Anthropomorphic/DBT/'

    DL_types = [args['model']]

    # DL_types = ['RED_DBT_PL4',
    #             'UNet2_DBT_PL4',
    #             'ResResNet_DBT_PL4',
    #             'RED_DBT_Noise2Sim',
    #             'UNet2_DBT_Noise2Sim',
    #             'ResResNet_DBT_Noise2Sim',
    #             'RED_DBT_VSTasLayer-MNSE_rnw0.02506266',
    #             'UNet2_DBT_VSTasLayer-MNSE_rnw0.03508772',
    #             'ResResNet_DBT_VSTasLayer-MNSE_rnw0.42105263',
    #             ]
    
    
    #%% Image parameters
    
    nRows = 1788    
    nCols = 1564 
    
    mAsFullDose = 60
    reducFactors = [args['rf']]

    n_all_fullDose = 20 - 1
    n_rlzs_fullDose = 10
    n_rlzs_GT = 10
    DL_ind = 0
           
    #%% Read clinical data
    
    myInv = lambda x : np.array((1./x[0],-x[1]/x[0]))
    
    nreducFactors = len(reducFactors)
    
    mnse_RD = np.empty(shape=(nreducFactors,3))
    resNoiseVar_RD = np.empty(shape=(nreducFactors,3))
    bias2_RD = np.empty(shape=(nreducFactors,3))
    
    mnse_MB = np.empty(shape=(nreducFactors,3))
    resNoiseVar_MB = np.empty(shape=(nreducFactors,3))
    bias2_MB = np.empty(shape=(nreducFactors,3))
    
    mnse_DL = np.empty(shape=(nreducFactors,3))
    resNoiseVar_DL = np.empty(shape=(nreducFactors,3))
    bias2_DL = np.empty(shape=(nreducFactors,3))
    
    # Read all full dose images
    fullDose_all = np.empty(shape=(nRows,nCols,15,n_all_fullDose))
    
    paths = sorted(list(pathlib.Path(path2Read + "31_" + str(mAsFullDose) ).glob('*/')))
    
    if paths == []:
        raise ValueError('No FD results found.')

    # Remove Raw_31_60_174911
    paths.pop(1)
    # paths.reverse()
    
    print('Reading FD images...')
    for idX, path in enumerate(paths):
        fullDose_all[:,:,:,idX] = readDicom(path,(nRows,nCols))
       
        
    # Generate the GT - fisrt time
    groundTruth = np.mean(fullDose_all[:,:,:,n_rlzs_GT:], axis=-1)
    # Generate a boolean mask for the breast
    maskBreast = groundTruth < 2500
    
    # for s in range(maskBreast.shape[-1]):
        # maskBreast[:,:,s] = binary_erosion(maskBreast[:,:,s],disk(39,dtype=bool))
    
    # Normalize the GT realizations
    for z in range(n_rlzs_GT, n_all_fullDose):
        for p in range(15):
            unique_rlzs = fullDose_all[:,:,p,z]
            unique_rlzs = np.polyval(myInv(np.polyfit(groundTruth[maskBreast[:,:,p]][:,p], unique_rlzs[maskBreast[:,:,p]], 1)), unique_rlzs)
            fullDose_all[:,:,p,z] = np.reshape(unique_rlzs, (nRows,nCols))
        
    # Generate again the GT after normalization
    groundTruth = np.mean(fullDose_all[:,:,:,n_rlzs_GT:], axis=-1)
    
    # Normalize the full dose realizations
    for z in range(n_rlzs_GT):
        for p in range(15):
            unique_rlzs = fullDose_all[:,:,p,z]
            unique_rlzs = np.polyval(myInv(np.polyfit(groundTruth[maskBreast[:,:,p]][:,p], unique_rlzs[maskBreast[:,:,p]], 1)), unique_rlzs)
            fullDose_all[:,:,p,z] = np.reshape(unique_rlzs, (nRows,nCols))
                  
    
    # Apply breask mask on both full dose anf GT realizations
    groundTruth_rlzs = fullDose_all[:,:,:,n_rlzs_GT:]
    fullDose_rlzs = fullDose_all[:,:,:,:n_rlzs_GT]

    mnse_FD_list, resNoiseVar_FD_list, bias2_FD_list = [], [], []
    for p in range(15):
        mnse_FD_tmp, resNoiseVar_FD_tmp, bias2_FD_tmp, _ = pyeval.MNSE(np.expand_dims(groundTruth_rlzs[:,:,p][maskBreast[:,:,p]], axis=0),
                                                                       np.expand_dims(fullDose_rlzs[:,:,p][maskBreast[:,:,p]], axis=0))
        mnse_FD_list.append(mnse_FD_tmp[0])
        resNoiseVar_FD_list.append(resNoiseVar_FD_tmp[0])
        bias2_FD_list.append(bias2_FD_tmp[0])

    mnse_FD = [np.mean(mnse_FD_list), np.std(mnse_FD_list, ddof=1), 0]
    resNoiseVar_FD = [np.mean(resNoiseVar_FD_list), np.std(resNoiseVar_FD_list, ddof=1), 0]
    bias2_FD = [np.mean(bias2_FD_list), np.std(bias2_FD_list, ddof=1), 0]

    df = pd.DataFrame(np.array(((["{:.7f} [{:.7f}, {:.7f}]".format(mnse_FD[0], mnse_FD[1], mnse_FD[2]),"{:.7f} [{:.7f}, {:.7f}]".format(resNoiseVar_FD[0], resNoiseVar_FD[1], resNoiseVar_FD[2]),"{:.7f} [{:.7f}, {:.7f}]".format(bias2_FD[0], bias2_FD[1], bias2_FD[2])]),
                                ),ndmin=2),
                                  columns=['Total MNSE', 'Residual-Noise', 'Bias-Squared'],
                                  index=["Full Dose"])

    df.to_csv(r'outputs.txt', sep=' ', header=None, mode='a')

    df = pd.DataFrame(np.array(((["{:.7f} [{:.7f}, {:.7f}]".format(mnse_FD[0], mnse_FD[1], mnse_FD[2]),"{:.7f} [{:.7f}, {:.7f}]".format(resNoiseVar_FD[0], resNoiseVar_FD[1], resNoiseVar_FD[2]),"{:.7f} [{:.7f}, {:.7f}]".format(bias2_FD[0], bias2_FD[1], bias2_FD[2])]),
                                ),ndmin=2),
                                  columns=['Total MNSE', 'Residual-Noise', 'Bias-Squared'],
                                  index=["Full Dose"])
            
    df.to_csv(r'outputs.txt', sep=' ', header=None, mode='a')
    
    del fullDose_all, fullDose_rlzs, mnse_FD_list, resNoiseVar_FD_list, bias2_FD_list
    
    # Read MB, restored results and reuced doses
    for reduc in reducFactors:
        
        mAsReducFactors = int((reduc / 100) * mAsFullDose)
        
        # Reduced doses
        paths = list(pathlib.Path(path2Read + "31_" + str(mAsReducFactors) ).glob('*'))
        if paths == []:
            raise ValueError('No RD results found.')
        paths = [path for path in paths if not 'MB' in str(path) and not 'DL' in str(path)]
        reduDose_rlzs = np.empty(shape=(nRows,nCols,15,n_rlzs_fullDose))
        for idX, path in enumerate(paths):
            all_rlzs =  readDicom(path,(nRows,nCols))
            for p in range(15):
                # Reduced doses
                unique_rlzs = all_rlzs[:,:,p]
                unique_rlzs = np.polyval(myInv(np.polyfit(groundTruth[maskBreast[:,:,p]][:,p], unique_rlzs[maskBreast[:,:,p]], 1)), unique_rlzs)
                reduDose_rlzs[:,:,p,idX] = np.reshape(unique_rlzs, (nRows,nCols))
                
        # reduDose[reduc] = reduDose_rlzs
    
    
        # MB restored doses
        paths = list(pathlib.Path(path2Read + "Restorations/31_" + str(mAsReducFactors) ).glob('MB*'))
        if paths == []:
            raise ValueError('No MB results found.')
        restDose_MB_rlzs = np.empty(shape=(nRows,nCols,15,n_rlzs_fullDose))
        for idX, path in enumerate(paths):
            all_rlzs =  readDicom(path,(nRows,nCols))
            for p in range(15):
                # MB restored doses
                unique_rlzs = all_rlzs[:,:,p]
                unique_rlzs = np.polyval(myInv(np.polyfit(groundTruth[maskBreast[:,:,p]][:,p], unique_rlzs[maskBreast[:,:,p]], 1)), unique_rlzs)
                restDose_MB_rlzs[:,:,p,idX] = np.reshape(unique_rlzs, (nRows,nCols))
            
        # restDose_MB[reduc] = restDose_MB_rlzs

        # Calculations for FD, MB and RD
        for idX, reduc in enumerate(reducFactors):

            mnse_RD_list, resNoiseVar_RD_list, bias2_RD_list = [], [], []

            for p in range(15):
                mnse_RD_tmp, resNoiseVar_RD_tmp, bias2_RD_tmp, _ = pyeval.MNSE(
                    np.expand_dims(groundTruth_rlzs[:,:,p][maskBreast[:, :, p]], axis=0),
                    np.expand_dims(reduDose_rlzs[:,:,p][maskBreast[:, :, p]], axis=0))
                mnse_RD_list.append(mnse_RD_tmp[0])
                resNoiseVar_RD_list.append(resNoiseVar_RD_tmp[0])
                bias2_RD_list.append(bias2_RD_tmp[0])

            mnse_RD = [np.mean(mnse_RD_list), np.std(mnse_RD_list, ddof=1), 0]
            resNoiseVar_RD = [np.mean(resNoiseVar_RD_list), np.std(resNoiseVar_RD_list, ddof=1), 0]
            bias2_RD = [np.mean(bias2_RD_list), np.std(bias2_RD_list, ddof=1), 0]

            mnse_MB_list, resNoiseVar_MB_list, bias2_MB_list = [], [], []

            for p in range(15):
                mnse_MB_tmp, resNoiseVar_MB_tmp, bias2_MB_tmp, _ = pyeval.MNSE(
                    np.expand_dims(groundTruth_rlzs[:,:,p][maskBreast[:, :, p]], axis=0),
                    np.expand_dims(restDose_MB_rlzs[:,:,p][maskBreast[:, :, p]], axis=0))
                mnse_MB_list.append(mnse_MB_tmp[0])
                resNoiseVar_MB_list.append(resNoiseVar_MB_tmp[0])
                bias2_MB_list.append(bias2_MB_tmp[0])

            mnse_MB = [np.mean(mnse_MB_list), np.std(mnse_MB_list, ddof=1), 0]
            resNoiseVar_MB = [np.mean(resNoiseVar_MB_list), np.std(resNoiseVar_MB_list, ddof=1), 0]
            bias2_MB = [np.mean(bias2_MB_list), np.std(bias2_MB_list, ddof=1), 0]

            # mnse_RD[idX,:], resNoiseVar_RD[idX,:], bias2_RD[idX,:], _= pyeval.MNSE(groundTruth_rlzs, np.expand_dims(reduDose_rlzs[maskBreast], axis=0))
            # mnse_MB[idX,:], resNoiseVar_MB[idX,:], bias2_MB[idX,:], _= pyeval.MNSE(groundTruth_rlzs, np.expand_dims(restDose_MB_rlzs[maskBreast], axis=0))

            df = pd.DataFrame(np.array(((["{:.7f} [{:.7f}, {:.7f}]".format(mnse_MB[0], mnse_MB[1], mnse_MB[2]),"{:.7f} [{:.7f}, {:.7f}]".format(resNoiseVar_MB[0], resNoiseVar_MB[1], resNoiseVar_MB[2]),"{:.7f} [{:.7f}, {:.7f}]".format(bias2_MB[0], bias2_MB[1], bias2_MB[2])]),
                                        (["{:.7f} [{:.7f}, {:.7f}]".format(mnse_RD[0], mnse_RD[1], mnse_RD[2]),"{:.7f} [{:.7f}, {:.7f}]".format(resNoiseVar_RD[0], resNoiseVar_RD[1], resNoiseVar_RD[2]),"{:.7f} [{:.7f}, {:.7f}]".format(bias2_RD[0], bias2_RD[1], bias2_RD[2])]),
                                        ),ndmin=2),
                              columns=['Total MNSE', 'Residual-Noise', 'Bias-Squared'],
                              index=["Model-Based","Noisy-{}mAs".format(mAsReducFactors)])
            
        df.to_csv(r'outputs.txt', sep=' ', header=None, mode='a')
        
    del reduDose_rlzs, restDose_MB_rlzs, unique_rlzs, all_rlzs
    
    
    # %% MNSE calculation
    
    
    # Loop through DL methods
    for indDL, DL_type in enumerate(DL_types):
        
        # Read DL 
        for idX, reduc in enumerate(reducFactors): 
            
            mAsReducFactors = int((reduc / 100) * mAsFullDose)
            
            print('Reading and calculating {}({}mAs) images...'.format(DL_type,mAsReducFactors))
            
            # DL restored doses
            paths = list(pathlib.Path(path2Read + "Restorations/31_" + str(mAsReducFactors) ).glob(DL_type + '*'))
            if paths == []:
                raise ValueError('No DL results found.')
            restDose_DL_rlzs = np.empty(shape=(nRows,nCols,15,n_rlzs_fullDose))
            for idZ, path in enumerate(paths):
                all_rlzs =  readDicom(path,(nRows,nCols))
                for p in range(15):            
                    # DL restored doses
                    unique_rlzs = all_rlzs[:,:,p]
                    unique_rlzs = np.polyval(myInv(np.polyfit(groundTruth[maskBreast[:,:,p]][:,p], unique_rlzs[maskBreast[:,:,p]], 1)), unique_rlzs)
                    restDose_DL_rlzs[:,:,p,idZ]  = np.reshape(unique_rlzs, (nRows,nCols))

            mnse_DL_list, resNoiseVar_DL_list, bias2_DL_list = [], [], []

            for p in range(15):
                mnse_DL_tmp, resNoiseVar_DL_tmp, bias2_DL_tmp, _ = pyeval.MNSE(
                    np.expand_dims(groundTruth_rlzs[:,:,p][maskBreast[:, :, p]], axis=0),
                    np.expand_dims(restDose_DL_rlzs[:,:,p][maskBreast[:, :, p]], axis=0))
                mnse_DL_list.append(mnse_DL_tmp[0])
                resNoiseVar_DL_list.append(resNoiseVar_DL_tmp[0])
                bias2_DL_list.append(bias2_DL_tmp[0])

            mnse_DL = [np.mean(mnse_DL_list), np.std(mnse_DL_list, ddof=1), 0]
            resNoiseVar_DL = [np.mean(resNoiseVar_DL_list), np.std(resNoiseVar_DL_list, ddof=1), 0]
            bias2_DL = [np.mean(bias2_DL_list), np.std(bias2_DL_list, ddof=1), 0]

            # mnse_DL[idX,:], resNoiseVar_DL[idX,:], bias2_DL[idX,:], _= pyeval.MNSE(groundTruth_rlzs, np.expand_dims(restDose_DL_rlzs[maskBreast], axis=0))

            
            df = pd.DataFrame(np.array(["{:.7f} [{:.7f}, {:.7f}]".format(mnse_DL[0], mnse_DL[1], mnse_DL[2]),
                                "{:.7f} [{:.7f}, {:.7f}]".format(resNoiseVar_DL[0], resNoiseVar_DL[1], resNoiseVar_DL[2]),
                                "{:.7f} [{:.7f}, {:.7f}]".format(bias2_DL[0], bias2_DL[1], bias2_DL[2])], ndmin=2),
                              columns=['Total MNSE', 'Residual-Noise', 'Bias-Squared'],
                              index=["DL-{}-{}mAs".format(DL_type,mAsReducFactors)])
            
            df.to_csv(r'outputs.txt', sep=' ', header=None, mode='a')
            # print(df)