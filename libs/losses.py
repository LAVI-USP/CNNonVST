#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:28:51 2022

@author: Rodrigo
"""

import torch


def MNSE(rlzs, groundTruth):
    # Crop borders
    # rlzs = rlzs[:, :, 10:-10, 10:-10]
    # groundTruth = groundTruth[:, :, 10:-10, 10:-10]

    n_batch = rlzs.shape[0] // 5  # Number of non-ground-truth realizations

    rlzs = torch.reshape(rlzs, (n_batch, 5, 1, rlzs.shape[-2], rlzs.shape[-1]))
    groundTruth = torch.reshape(groundTruth, (n_batch, 5, 1, rlzs.shape[-2], rlzs.shape[-1]))

    resNoiseVar = torch.mean(torch.var(rlzs, axis=1, keepdim=True) / groundTruth)

    bias2 = torch.mean(((torch.mean(rlzs, axis=1, keepdim=True) - groundTruth) ** 2) / groundTruth)

    factor2 = (resNoiseVar / 5)

    bias2 = bias2 - factor2

    return bias2, resNoiseVar
