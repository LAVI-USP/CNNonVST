#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 10:01:27 2022

@author: rodrigo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import pydicom
import pathlib
from scipy.io import loadmat


# %%

def calc_digital_ps(I, n, px=1, use_window=0, average_stack=0, use_mean=0):
    '''
    
    Description: Calculates the digital power spectrum (PS) realizations.
    
    Input:
        - I = stack of ROIs
        - n = n-dimensional noise realizations, e.g. 2
        - px = pixel/detector size
        - use_window = Useful for avoiding spectral leakage?
        - average_stack = mean on all ROIs?
        - use_mean = subtract mean or not?
    
    Output:
        - nps = noise-power spectrum (NPS)
        - f = frequency vector
            
    
    -----------------
    
    
    '''

    size_I = I.shape

    if size_I[0] != size_I[1]:
        raise ValueError("ROI must be symmetric.")

    roi_size = size_I[0]

    # Cartesian coordinates
    x = np.linspace(-roi_size / 2, roi_size / 2, roi_size)
    _, x = np.meshgrid(x, x)

    # frequency vector
    f = np.linspace(-0.5, 0.5, roi_size) / px

    # radial coordinates
    r = np.sqrt(x ** 2 + np.transpose(x) ** 2)

    # Hann window to avoid spectral leakage
    if use_window:
        hann = 0.5 * (1 + np.cos(np.pi * r / (roi_size / 2)))
        hann[r > roi_size / 2] = 0
        hann = np.expand_dims(hann, axis=-1)
    else:
        hann = 1

    # detrending by subtracting the mean of each ROI
    # more advanced schemes include subtracting a surface, but that is
    # currently not included
    if use_mean:
        S = np.mean(I, axis=(0, 1))
        S = np.expand_dims(np.expand_dims(S, axis=0), axis=0)
    else:
        S = 0

    F = (I - S) * hann

    # equivalent to fftn
    F = np.fft.fftshift(np.fft.fft2(F, axes=(0, 1)))

    # PS
    ps = np.abs(F) ** 2  # / roi_size ** n * px ** n / (np.sum(hann**2) / hann.size)

    # averaging the NPS over the ROIs assuming ergodicity
    if average_stack:
        ps = np.mean(ps, axis=2)

    ps = ((px ** 2) / (size_I[0] ** 2)) * ps

    return ps, f


def powerSpectrum(img, roiSize=[], pixelSize=[]):
    '''
    
    Description: Calculates the digital power spectrum (PS).
    
    Input:
        - img = image to calculate NPS
        - roiSize = Size of ROI that will be extracted
        - pixelSize = pixel/detector size
    
    Output:
        - nps2D = 2D PS
        - nps1D = 1D PS (radial)
        - f1D = frequency vector
        
    '''

    img = img.astype('float64')

    M, N = img.shape

    if not roiSize:
        roiSize = M

    if roiSize <= 0 or roiSize > M:
        roiSize = M

    if not pixelSize:
        raise ValueError("No pixel size input")

    rois = []
    means = []

    for i in range(0, M - roiSize, roiSize):
        for j in range(0, N - roiSize, roiSize):
            roi = img[i:i + roiSize, j:j + roiSize]

            if np.sum(roi == 0) < 1:
                means.append(roi.mean())
                roi -= means[-1]
                rois.append(roi)

    rois = np.stack(rois, axis=-1)

    # NPS 2D
    nps2D, _ = calc_digital_ps(rois, 2, pixelSize, 1, 1, 0)

    # Normalization (consireding segmented img)
    nps2D /= np.mean(means) ** 2  # img[img>0].mean()** 2

    # NPS 1D - RADIAL - Euclidean Distance
    cx = roiSize // 2

    nFreqSample = cx + 1
    nyquist = 1 / (2 * pixelSize)

    # Distance matrix (u, v) plane
    x = np.arange(-cx, roiSize - cx)
    xx, yy = np.meshgrid(x, x)
    radialDst = np.round(np.sqrt(xx ** 2 + yy ** 2))

    # Generate 1D NPS
    nps1D = np.empty(shape=(nFreqSample))
    for k in range(nFreqSample):
        nps1D[k] = nps2D[radialDst == k].mean()

    f1D = np.linspace(0, nyquist, nFreqSample)

    return nps2D, nps1D, f1D


# Read Dicom function
def readDicom(dir2Read, imgSize):
    # List dicom files
    dcmFiles = list(pathlib.Path(dir2Read).glob('*.dcm'))

    dcmImg = np.empty([imgSize[0], imgSize[1]])

    if not dcmFiles:
        raise ValueError('No DICOM files found in the specified path.')

    for dcm in dcmFiles:

        ind = int(str(dcm).split('/')[-1].split('_')[-1].split('.')[0])

        if ind == 7:
            dcmH = pydicom.dcmread(str(dcm))

            dcmImg = dcmH.pixel_array[130:-130, 50:-50].astype('float32')  # [0:,1128:].astype('float32')

    return dcmImg


# %% General parameters

path2Read = '/media/rodrigo/Dados_2TB/Imagens/UPenn/Phantom/Anthropomorphic/DBT/'

DL_types = ['UNet2_DBT_VSTasLayer-MNSE_rnw0.03508772_50_Raw_31_30_180230',
            'UNet2_DBT_PL4_50_Raw_31_30_180230',
            'UNet2_DBT_Noise2Sim_50_Raw_31_30_180230',
            'ResResNet_DBT_VSTasLayer-MNSE_rnw0.42105263_50_Raw_31_30_180230',
            'ResResNet_DBT_PL4_50_Raw_31_30_180230',
            'ResResNet_DBT_Noise2Sim_50_Raw_31_30_180230',
            'RED_DBT_VSTasLayer-MNSE_rnw0.02506266_50_Raw_31_30_180230',
            'RED_DBT_PL4_50_Raw_31_30_180230',
            'RED_DBT_Noise2Sim_50_Raw_31_30_180230']

DL_names = ['UNet-$VST_{0.035}$',
            'UNet-PL4',
            'UNet-Noise2Sim',
            'ResNet-$VST_{0.421}$',
            'ResNet-PL4',
            'ResNet-Noise2Sim',
            'RED-$VST_{0.025}$',
            'RED-PL4',
            'RED-Noise2Sim']

# %% Image parameters

nRows = 1788
nCols = 1564

mAsFullDose = 60
reducFactors = [50]

n_all_fullDose = 20
n_rlzs_fullDose = 10
n_rlzs_GT = 10
DL_ind = 0

roiSize = 50

# %% Read clinical data

nreducFactors = len(reducFactors)

# Read all full dose images
fullDose_all = np.empty(shape=(nRows, nCols, n_all_fullDose))

paths = sorted(list(pathlib.Path(path2Read + "31_" + str(mAsFullDose)).glob('*/')))

if paths == []:
    raise ValueError('No FD results found.')

print('Reading FD images...')
for idX, path in enumerate(paths):
    fullDose_all[:, :, idX] = readDicom(path, (nRows, nCols))

# Generate the GT - fisrt time
groundTruth = np.mean(fullDose_all[:, :, n_rlzs_GT:], axis=-1)
# Generate a boolean mask for the breast
maskBreast = groundTruth < 2500

ps_FD = np.zeros((26))

for z in range(10):
    _, nps1D, f1D = powerSpectrum(fullDose_all[:, :, z] * maskBreast, roiSize=roiSize, pixelSize=0.14)
    ps_FD += nps1D

ps_FD /= 10

ind = np.where(f1D > 1)  # np.where((f1D > 0.1) & (f1D < 7))

ax = []
figs = []
for k in range(len(reducFactors)):
    figs.append(plt.figure())
    ax.append(figs[-1].add_subplot(1, 1, 1))

for k in range(len(reducFactors)):
    ax[k].plot(f1D[ind], ps_FD[ind], label=r'FD', marker='o')

del fullDose_all
# %%

# Read MB, restored results and reuced doses
for k, reduc in enumerate(reducFactors):

    mAsReducFactors = int((reduc / 100) * mAsFullDose)

    # Reduced doses
    paths = list(pathlib.Path(path2Read + "31_" + str(mAsReducFactors)).glob('*'))
    if paths == []:
        raise ValueError('No RD results found.')
    paths = [path for path in paths if not 'MB' in str(path) and not 'DL' in str(path)]
    reduDose_rlzs = np.empty(shape=(nRows, nCols, n_rlzs_fullDose))
    for idX, path in enumerate(paths):
        reduDose_rlzs[:, :, idX] = readDicom(path, (nRows, nCols))

    ps_RD = np.zeros((26))

    for z in range(1):
        _, nps1D, f1D = powerSpectrum(reduDose_rlzs[:, :, z] * maskBreast, roiSize=roiSize, pixelSize=0.14)
        ps_RD += nps1D

    ps_RD /= 1

    ax[k].plot(f1D[ind], ps_RD[ind], label=r'LD')

    # MB restored doses
    paths = list(pathlib.Path(path2Read + "Restorations/31_" + str(mAsReducFactors)).glob('MB*'))
    if paths == []:
        raise ValueError('No MB results found.')
    restDose_MB_rlzs = np.empty(shape=(nRows, nCols, n_rlzs_fullDose))
    for idX, path in enumerate(paths):
        restDose_MB_rlzs[:, :, idX] = readDicom(path, (nRows, nCols))

    ps_MB = np.zeros((26))

    for z in range(1):
        _, nps1D, f1D = powerSpectrum(restDose_MB_rlzs[:, :, z] * maskBreast, roiSize=roiSize, pixelSize=0.14)
        ps_MB += nps1D

    ps_MB /= 1

    ax[k].plot(f1D[ind], ps_MB[ind], label=r'MB')

del reduDose_rlzs, restDose_MB_rlzs

# %%

# Loop through DL methods
for indDL, (DL_type, DL_name) in enumerate(zip(DL_types, DL_names)):

    # Read DL 
    for idX, reduc in enumerate(reducFactors):

        mAsReducFactors = int((reduc / 100) * mAsFullDose)

        print('Reading and calculating {}({}mAs) images...'.format(DL_type, mAsReducFactors))

        # DL restored doses
        paths = list(pathlib.Path(path2Read + "Restorations/31_" + str(mAsReducFactors)).glob(DL_type + '*'))
        if paths == []:
            raise ValueError('No DL results found.')
        restDose_DL_rlzs = np.empty(shape=(nRows, nCols, n_rlzs_fullDose))
        for idZ, path in enumerate(paths):
            restDose_DL_rlzs[:, :, idZ] = readDicom(path, (nRows, nCols))

        ps_DL = np.zeros((26))

        for z in range(1):
            _, nps1D, f1D = powerSpectrum(restDose_DL_rlzs[:, :, z] * maskBreast, roiSize=roiSize, pixelSize=0.14)
            ps_DL += nps1D

        ps_DL /= 1

        ax[idX].plot(f1D[ind], ps_DL[ind], label=r'{}'.format(DL_name))

# %%

for idX, reduc in enumerate(reducFactors):
    ax[idX].set_yscale('log')
    # ax[idX].set_xscale('log')
    ax[idX].set_xlabel(r'Spatial Freq. ($mm^{-1}$)', fontsize=16)
    ax[idX].set_ylabel('Power Spectral Density ($mm^2$)', fontsize=16)
    ax[idX].legend()
    # ax[idX].set_title('{:d}%'.format(reduc))

    figs[idX].tight_layout()
    figs[idX].savefig('PS-{:02d}.png'.format(reduc))
