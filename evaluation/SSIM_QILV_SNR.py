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
import argparse
import sys
import cv2
import scipy.stats as st

from scipy.signal import fftconvolve
from skimage.metrics import structural_similarity as ssim


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """

    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def quality_index_local_variance(I, I2, Ws, raw=False, mask=None):


    I = I.astype(np.float64)
    I2 = I2.astype(np.float64)

    # Data type maximum range
    L = 2302  # 4095

    # return 0

    K = [0.01, 0.03]
    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2

    if Ws == 0:
        window = fspecial_gauss(11, 1.5)
    else:
        window = fspecial_gauss(Ws, 1.5)

    window = window / np.sum(window)

    # Local means
    M1 = fftconvolve(I, window, mode='same')
    M2 = fftconvolve(I2, window, mode='same')

    # Local Variances
    V1 = fftconvolve(I ** 2, window, mode='same') - M1 ** 2
    V2 = fftconvolve(I2 ** 2, window, mode='same') - M2 ** 2

    if mask is not None:

        # Global statistics:
        m1 = np.mean(V1[mask])
        m2 = np.mean(V2[mask])
        s1 = np.std(V1[mask], ddof=1)
        s2 = np.std(V2[mask], ddof=1)
        s12 = np.mean((V1[mask] - m1) * (V2[mask] - m2))

    else:
        # Global statistics:
        m1 = np.mean(V1)
        m2 = np.mean(V2)
        s1 = np.std(V1, ddof=1)
        s2 = np.std(V2, ddof=1)
        s12 = np.mean((V1 - m1) * (V2 - m2))

    # Index
    ind1 = ((2 * m1 * m2 + C1) / (m1 ** 2 + m2 ** 2 + C1))
    ind2 = (2 * s1 * s2 + C2) / (s1 ** 2 + s2 ** 2 + C2)
    ind3 = (s12 + C2 / 2) / (s1 * s2 + C2 / 2)
    ind = ind1 * ind2 * ind3

    return ind


def readDicom(dir2Read, imgSize):
    # List dicom files
    dcmFiles = list(pathlib.Path(dir2Read).glob('*.dcm'))

    dcmImg = np.empty([imgSize[0], imgSize[1], len(dcmFiles)])

    if not dcmFiles:
        raise ValueError('No DICOM files found in the specified path.')

    for dcm in dcmFiles:
        dcmH = pydicom.dcmread(str(dcm))

        ind = int(str(dcm).split('/')[-1].split('_')[-1].split('.')[0])

        dcmImg[:, :, ind] = dcmH.pixel_array[130:-130, 50:-50].astype(np.float64)

    return dcmImg


def myBlur(x):
    for p in range(15):
        x[..., p] = cv2.blur(x[..., p], (15, 15), borderType=cv2.BORDER_REFLECT)

    return x


# %%

if __name__ == '__main__':

    ap = argparse.ArgumentParser(description='Restore low-dose mamography')
    ap.add_argument("--rf", type=int, default=50, required=True,
                    help="Reduction factor in percentage. (default: 50)")
    ap.add_argument("--model", type=str, required=True,
                    help="Model type")

    sys.argv = sys.argv + ['--rf', '50', '--model',
                           'RED_DBT_PL4']  # VSTasLayer-MNSE_rnw0.02506266']  # VSTasLayer-MNSE_ep_1rnw0.0']

    args = vars(ap.parse_args())

    model_type = args['model']

    # %% General parameters

    # path2Read = '/home/laviusp/Documents/Rodrigo_Vimieiro/phantom/'
    path2Read = '/media/rodrigo/Dados_2TB/Imagens/UPenn/Phantom/Anthropomorphic/DBT/'

    DL_types = [args['model']]

    DL_types = ['RED_DBT_PL4',
                'UNet2_DBT_PL4',
                'ResResNet_DBT_PL4',
                'RED_DBT_Noise2Sim',
                'UNet2_DBT_Noise2Sim',
                'ResResNet_DBT_Noise2Sim',
                'RED_DBT_VSTasLayer-MNSE_rnw0.02506266',
                'UNet2_DBT_VSTasLayer-MNSE_rnw0.03508772',
                'ResResNet_DBT_VSTasLayer-MNSE_rnw0.42105263',
                ]

    # %% Image parameters

    nRows = 1788
    nCols = 1564

    mAsFullDose = 60
    reducFactors = [args['rf']]

    n_all_fullDose = 20 - 1
    n_rlzs_fullDose = 10
    n_rlzs_GT = 10
    DL_ind = 0

    # %% Read clinical data

    myInv = lambda x: np.array((1. / x[0], -x[1] / x[0]))

    nreducFactors = len(reducFactors)

    # Read all full dose images
    fullDose_all = np.empty(shape=(nRows, nCols, 15, n_all_fullDose))

    paths = sorted(list(pathlib.Path(path2Read + "31_" + str(mAsFullDose)).glob('*/')))

    if paths == []:
        raise ValueError('No FD results found.')

    # Remove Raw_31_60_174911
    paths.pop(1)
    # paths.reverse()

    print('Reading FD images...')
    for idX, path in enumerate(paths):
        fullDose_all[:, :, :, idX] = readDicom(path, (nRows, nCols))

    # Generate the GT - fisrt time
    groundTruth = np.mean(fullDose_all[:, :, :, n_rlzs_GT:], axis=-1)
    # Generate a boolean mask for the breast
    maskBreast = groundTruth < 2500

    # Normalize the GT realizations
    for z in range(n_rlzs_GT, n_all_fullDose):
        for p in range(15):
            unique_rlzs = fullDose_all[:, :, p, z]
            unique_rlzs = np.polyval(
                myInv(np.polyfit(groundTruth[maskBreast[:, :, p]][:, p], unique_rlzs[maskBreast[:, :, p]], 1)),
                unique_rlzs)
            fullDose_all[:, :, p, z] = np.reshape(unique_rlzs, (nRows, nCols))

    # Generate again the GT after normalization
    groundTruth = np.mean(fullDose_all[:, :, :, n_rlzs_GT:], axis=-1)

    # Normalize the full dose realizations
    for z in range(n_rlzs_GT):
        for p in range(15):
            unique_rlzs = fullDose_all[:, :, p, z]
            unique_rlzs = np.polyval(
                myInv(np.polyfit(groundTruth[maskBreast[:, :, p]][:, p], unique_rlzs[maskBreast[:, :, p]], 1)),
                unique_rlzs)
            fullDose_all[:, :, p, z] = np.reshape(unique_rlzs, (nRows, nCols))

    # print(fullDose_all[maskBreast].min(), fullDose_all[maskBreast].max())

    std_FD = np.sqrt(np.var(fullDose_all, ddof=1, axis=-1))
    mean_FD = np.mean(fullDose_all, axis=-1)
    mask_std = np.where(std_FD == 0, 1, 0)

    ssim_vals = []
    for z in range(n_rlzs_GT):
        for p in range(15):
            ssim_vals.append(ssim(groundTruth[..., p][maskBreast[..., p]], fullDose_all[:, :, p, z][maskBreast[..., p]],
                                  data_range=2302))

    with open(r'outputs_ssim.txt', "a") as file:
        file.write("FD - {} +- {}\n".format(np.mean(ssim_vals), np.std(ssim_vals, ddof=1)))

    # print("SSIM - DL-{}-{}mAs - {}\n".format(DL_type,
    #                                          mAsReducFactors,
    #                                          np.mean(ssim_vals)))

    qilv_vals = []
    for z in range(n_rlzs_GT):
        for p in range(15):
            qilv_vals.append(quality_index_local_variance(fullDose_all[:, :, p, z],
                                                          groundTruth[..., p],
                                                          Ws=0,
                                                          raw=False,
                                                          mask=maskBreast[..., p]))

    with open(r'outputs_qilv.txt', "a") as file:
        file.write("FD - {} +- {}\n".format(np.mean(qilv_vals),
                                            np.std(qilv_vals, ddof=1)))

    # print("QILV - DL-{}-{}mAs - {}\n".format(DL_type,
    #                                          mAsReducFactors,
    #                                          np.mean(qilv_vals)))

    del fullDose_all

    # Read MB, restored results and reuced doses
    for reduc in reducFactors:
        mAsReducFactors = int((reduc / 100) * mAsFullDose)

        # Reduced doses
        paths = list(pathlib.Path(path2Read + "31_" + str(mAsReducFactors)).glob('*'))
        if paths == []:
            raise ValueError('No RD results found.')
        paths = [path for path in paths if not 'MB' in str(path) and not 'DL' in str(path)]
        reduDose_rlzs = np.empty(shape=(nRows, nCols, 15, n_rlzs_fullDose))
        for idX, path in enumerate(paths):
            all_rlzs = readDicom(path, (nRows, nCols))
            for p in range(15):
                # Reduced doses
                unique_rlzs = all_rlzs[:, :, p]
                unique_rlzs = np.polyval(
                    myInv(np.polyfit(groundTruth[maskBreast[:, :, p]][:, p], unique_rlzs[maskBreast[:, :, p]], 1)),
                    unique_rlzs)
                reduDose_rlzs[:, :, p, idX] = np.reshape(unique_rlzs, (nRows, nCols))

        # print(reduDose_rlzs[maskBreast].min(), reduDose_rlzs[maskBreast].max())

        ssim_vals = []
        for idZ, _ in enumerate(paths):
            for p in range(15):
                ssim_vals.append(ssim(groundTruth[..., p][maskBreast[..., p]], reduDose_rlzs[:,:,p, idZ][maskBreast[..., p]], data_range=2302))

        with open(r'outputs_ssim.txt', "a") as file:
            file.write("LD-{}mAs - {} +- {}\n".format(mAsReducFactors,
                                                      np.mean(ssim_vals),
                                                      np.std(ssim_vals, ddof=1)))

        qilv_vals = []
        for z in range(n_rlzs_GT):
            for p in range(15):
                qilv_vals.append(quality_index_local_variance(reduDose_rlzs[:, :, p, z],
                                                              groundTruth[:, :, p],
                                                              Ws=0,
                                                              raw=False,
                                                              mask=maskBreast[:, :, p]))

        with open(r'outputs_qilv.txt', "a") as file:
            file.write("LD-{}mAs - {} +- {}\n".format(mAsReducFactors,
                                                      np.mean(qilv_vals),
                                                      np.std(qilv_vals, ddof=1)))

        # reduDose[reduc] = reduDose_rlzs

        # MB restored doses
        paths = list(pathlib.Path(path2Read + "Restorations/31_" + str(mAsReducFactors)).glob('MB*'))
        if paths == []:
            raise ValueError('No MB results found.')
        restDose_MB_rlzs = np.empty(shape=(nRows, nCols, 15, n_rlzs_fullDose))
        for idX, path in enumerate(paths):
            all_rlzs = readDicom(path, (nRows, nCols))
            for p in range(15):
                # MB restored doses
                unique_rlzs = all_rlzs[:, :, p]
                unique_rlzs = np.polyval(
                    myInv(np.polyfit(groundTruth[maskBreast[:, :, p]][:, p], unique_rlzs[maskBreast[:, :, p]], 1)),
                    unique_rlzs)
                restDose_MB_rlzs[:, :, p, idX] = np.reshape(unique_rlzs, (nRows, nCols))

        # print(restDose_MB_rlzs[maskBreast].min(), restDose_MB_rlzs[maskBreast].max())

        ssim_vals = []
        for idZ, _ in enumerate(paths):
            for p in range(15):
                ssim_vals.append(ssim(groundTruth[..., p][maskBreast[..., p]], restDose_MB_rlzs[:,:,p, idZ][maskBreast[..., p]], data_range=2302))

        with open(r'outputs_ssim.txt', "a") as file:
            file.write("MB-{}mAs - {} +- {}\n".format(mAsReducFactors,
                                                      np.mean(ssim_vals),
                                                      np.std(ssim_vals, ddof=1)))

        qilv_vals = []
        for z in range(n_rlzs_GT):
            for p in range(15):
                qilv_vals.append(quality_index_local_variance(restDose_MB_rlzs[:, :, p, z],
                                                              groundTruth[:, :, p],
                                                              Ws=0,
                                                              raw=False,
                                                              mask=maskBreast[:, :, p]))

        with open(r'outputs_qilv.txt', "a") as file:
            file.write("MB-{}mAs - {} +- {}\n".format(mAsReducFactors,
                                                      np.mean(qilv_vals),
                                                      np.std(qilv_vals, ddof=1)))

        # restDose_MB[reduc] = restDose_MB_rlzs

        # Calculations for FD, MB and RD
        reduDose_std = np.sqrt(np.var(reduDose_rlzs, ddof=1, axis=-1))
        reduDose_mean = np.mean(reduDose_rlzs, axis=-1)
        mask_std |= np.where(reduDose_std == 0, 1, 0)

        restDose_MB_std = np.sqrt(np.var(restDose_MB_rlzs, ddof=1, axis=-1))
        restDose_MB_mean = np.mean(restDose_MB_rlzs, axis=-1)
        mask_std |= np.where(restDose_MB_std == 0, 1, 0)


    del reduDose_rlzs, restDose_MB_rlzs, unique_rlzs, all_rlzs

    # %% MNSE calculation

    # Loop through DL methods
    for indDL, DL_type in enumerate(DL_types):

        # Read DL
        for idX, reduc in enumerate(reducFactors):

            mAsReducFactors = int((reduc / 100) * mAsFullDose)

            print('Reading and calculating {}({}mAs) images...'.format(DL_type, mAsReducFactors))

            # DL restored doses
            paths = list(pathlib.Path(path2Read + "Restorations/31_" + str(mAsReducFactors)).glob(DL_type + '*'))
            if paths == []:
                raise ValueError('No DL results found.')
            restDose_DL_rlzs = np.empty(shape=(nRows, nCols, 15, n_rlzs_fullDose))
            for idZ, path in enumerate(paths):
                all_rlzs = readDicom(path, (nRows, nCols))
                for p in range(15):
                    # DL restored doses
                    unique_rlzs = all_rlzs[:, :, p]
                    unique_rlzs = np.polyval(
                        myInv(np.polyfit(groundTruth[maskBreast[:, :, p]][:, p], unique_rlzs[maskBreast[:, :, p]], 1)),
                        unique_rlzs)
                    restDose_DL_rlzs[:, :, p, idZ] = np.reshape(unique_rlzs, (nRows, nCols))

            # print(restDose_DL_rlzs[maskBreast].min(), restDose_DL_rlzs[maskBreast].max())

            restDose_DL_std = np.sqrt(np.var(restDose_DL_rlzs, ddof=1, axis=-1))
            restDose_DL_mean = np.mean(restDose_DL_rlzs, axis=-1)
            # mask_std |= np.where(restDose_DL_std == 0, 1, 0)

        # del all_rlzs, unique_rlzs

        # mask_std = np.where(mask_std > 0, 0, 1).astype('bool')

        mask_std = maskBreast #& mask_std

        snr_FD_map = myBlur(mean_FD) / myBlur(std_FD)
        snr_FD = []
        for p in range(15):
            snr_FD.append(np.mean(snr_FD_map[:, :, p][mask_std[:, :, p]]))

        snr_FD_CI = np.std(snr_FD, ddof=1), 0
        snr_FD = np.mean(snr_FD)

        # snr_FD_CI = st.t.interval(0.95, snr_FD_map[mask_std].shape[0] - 1,
        #                           loc=np.mean(snr_FD_map[mask_std]),
        #                           scale=st.sem(snr_FD_map[mask_std], axis=-1))


        with open(r'outputs_snr.txt', "a") as file:
            file.write("FD-{}mAs - {} [{},{}]\n".format(mAsReducFactors,
                                                        snr_FD,
                                                        snr_FD_CI[0],
                                                        snr_FD_CI[1]))

        # del mean_FD, std_FD, snr_FD_map

        snr_2D = myBlur(reduDose_mean) / myBlur(reduDose_std)
        snr_RD = []
        for p in range(15):
            snr_RD.append(np.mean(snr_2D[:, :, p][mask_std[:, :, p]]))

        snr_RD_CI = np.std(snr_RD, ddof=1), 0
        snr_RD = np.mean(snr_RD)

        # snr_RD_CI = st.t.interval(0.95, snr_2D[mask_std].shape[0] - 1,
        #                                 loc=np.mean(snr_2D[mask_std]),
        #                                 scale=st.sem(snr_2D[mask_std], axis=-1))

        with open(r'outputs_snr.txt', "a") as file:
            file.write("LD-{}mAs - {} [{},{}]\n".format(mAsReducFactors,
                                                        snr_RD,
                                                        snr_RD_CI[0],
                                                        snr_RD_CI[1]))

        # del reduDose_mean, reduDose_std

        snr_2D = myBlur(restDose_MB_mean) / myBlur(restDose_MB_std)
        snr_MB = []
        for p in range(15):
            snr_MB.append(np.mean(snr_2D[:, :, p][mask_std[:, :, p]]))

        snr_MB_CI = np.std(snr_MB, ddof=1), 0
        snr_MB = np.mean(snr_MB)
        # snr_MB_CI = st.t.interval(0.95, snr_2D[mask_std].shape[0] - 1,
        #                                 loc=np.mean(snr_2D[mask_std]),
        #                                 scale=st.sem(snr_2D[mask_std], axis=-1))

        with open(r'outputs_snr.txt', "a") as file:
            file.write("MB-{}mAs - {} [{},{}]\n".format(mAsReducFactors,
                                                        snr_MB,
                                                        snr_MB_CI[0],
                                                        snr_MB_CI[1]))

        # del restDose_MB_mean, restDose_MB_std, snr_2D

        snr_2D = myBlur(restDose_DL_mean) / myBlur(restDose_DL_std)
        snr_DL = []
        for p in range(15):
            snr_DL.append(np.mean(snr_2D[:, :, p][mask_std[:, :, p]]))

        snr_DL_CI = np.std(snr_DL, ddof=1), 0
        snr_DL = np.mean(snr_DL)
        # snr_DL_CI = st.t.interval(0.95, snr_2D[mask_std].shape[0] - 1,
        #                           loc=np.mean(snr_2D[mask_std]),
        #                           scale=st.sem(snr_2D[mask_std], axis=-1))

        with open(r'outputs_snr.txt', "a") as file:
            file.write("DL-{}-{}mAs - {} [{},{}]\n".format(DL_type,
                                                           mAsReducFactors,
                                                           snr_DL,
                                                           snr_DL_CI[0],
                                                           snr_DL_CI[1]))

        # del mask_std, restDose_DL_mean, restDose_DL_std

        ssim_vals = []
        for idZ, _ in enumerate(paths):
            for p in range(15):
                ssim_vals.append(ssim(groundTruth[..., p][maskBreast[..., p]], restDose_DL_rlzs[:,:,p, idZ][maskBreast[..., p]], data_range=2302))

        with open(r'outputs_ssim.txt', "a") as file:
            file.write("DL-{}-{}mAs - {} +- {}\n".format(DL_type,
                                                         mAsReducFactors,
                                                         np.mean(ssim_vals),
                                                         np.std(ssim_vals, ddof=1)))

        qilv_vals = []
        for z in range(n_rlzs_GT):
            for p in range(15):
                qilv_vals.append(quality_index_local_variance(restDose_DL_rlzs[:, :, p, z],
                                                              groundTruth[:, :, p],
                                                              Ws=0,
                                                              raw=False,
                                                              mask=maskBreast[:, :, p]))

        with open(r'outputs_qilv.txt', "a") as file:
            file.write("DL-{}-{}mAs - {} +- {}\n".format(DL_type,
                                                         mAsReducFactors,
                                                         np.mean(qilv_vals),
                                                         np.std(qilv_vals, ddof=1)))
