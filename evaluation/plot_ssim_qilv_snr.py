#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 16:27:26 2022

@author: rodrigo
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


# %%

def extract_info_qilv(file_name):

    f = open(file_name + ".txt", "r")
    wrn_mnse_lines = f.read().split('DL')
    f.close()

    wrn_mnse_lines = ['DL' + x for x in wrn_mnse_lines][1:]

    metric_dict = {}
    metric_dict['RED'] = {}
    metric_dict['UNet2'] = {}
    metric_dict['ResResNet'] = {}
    # metric_dict['ResNet'] = {}

    for wrn_mnse_line in wrn_mnse_lines:
        split_values = wrn_mnse_line.split(' ')
        if split_values[0] != '':
            start_index = split_values[0].find('DL-') + len('DL-')
            end_index = split_values[0].find('_DBT')
            dl_model = split_values[0][start_index:end_index]

            start_index = split_values[0].find('rnw') + len('rnw')
            end_index = split_values[0].find('-30mAs')
            rnw = float(split_values[0][start_index:end_index])

            metric_value = split_values[2]

            metric_dict[dl_model][rnw] = {'metric': float(metric_value)}

    return metric_dict


def extract_info_ssim(file_name):

    f = open(file_name + ".txt", "r")
    wrn_mnse_lines = f.read().split('\n')
    f.close()

    metric_dict = {}
    metric_dict['RED'] = {}
    metric_dict['UNet2'] = {}
    metric_dict['ResResNet'] = {}
    # metric_dict['ResNet'] = {}

    for wrn_mnse_line in wrn_mnse_lines:
        split_values = wrn_mnse_line.split(' ')
        if split_values[0] != '':
            start_index = split_values[0].find('DL-') + len('DL-')
            end_index = split_values[0].find('_DBT')
            dl_model = split_values[0][start_index:end_index]

            start_index = split_values[0].find('rnw') + len('rnw')
            end_index = split_values[0].find('-30mAs')
            rnw = float(split_values[0][start_index:end_index])

            metric_value = split_values[1]

            metric_dict[dl_model][rnw] = {'metric': float(metric_value)}

    return metric_dict


# %%
if __name__ == '__main__':

    path2read = ".."
    path2write = '/home/rodrigo/Dropbox/'

    # %%

    metrics = ['ssim', 'snr', 'qilv']
    # ylims = [[0.9989, 0.99935], []]

    for metric in metrics:

        if metric == 'ssim':
            metric_dict = extract_info_ssim("{}/outputs_{}".format(path2read, metric))
        else:
            metric_dict = extract_info_qilv("{}/outputs_{}".format(path2read, metric))

        # %% Plotando com epoch em fig diferente
        
        for idx, model_key in enumerate(metric_dict):

            wrn_all = np.array(list(metric_dict[model_key].keys()))
            metric_ep = np.array([metric_dict[model_key][key]['metric'] for key in metric_dict[model_key]])

            print("{}-{} has {} wrn values;".format(model_key, metric, wrn_all.shape[0]))

            if idx==0:
                legends = []
                fig, ax1 = plt.subplots()

            ax1.set_ylabel(metric.upper(), fontsize=16)
            ax1.set_xlabel("$\lambda_{RN}$", fontsize=16)
            lns, = ax1.plot(wrn_all, metric_ep, '.', label=model_key)
            legends.append(lns)
            # lns3, = ax1.plot(np.linspace(np.min(wrn_all), np.max(wrn_all), num=wrn_all.shape[0]),
            #                  np.ones_like(wrn_all) * 0.1186, '--', color='lightblue', label='Full-dose RN', linewidth=2)
            # ax1.set_ylim(0.10, 0.24)
            # ax1.set_xlim(0.0, 0.5)

            # for k in range(wrn_all.shape[0]):
            #     ax1.plot(np.array((wrn_all[k], wrn_all[k])),
            #               np.array((0.0023, 0.0040)),
            #               '--',
            #               color='black',
            #               alpha=.2)

        ax1.grid()
        ax1.legend(handles=legends, prop={'size': 9})#, loc='upper center')
        # ax1.set_title(model_key)

        fig.tight_layout()
        fig.savefig(path2write + metric + "_wRNxb2xRN_ep02.png", bbox_inches="tight")





