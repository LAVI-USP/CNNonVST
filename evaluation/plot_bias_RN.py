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

def extract_bias_rn(file_name):
    f = open(file_name + ".txt", "r")
    wrn_mnse_lines = f.read().split('\n')
    f.close()

    mnse_dict = {}
    mnse_dict['RED'] = {}
    mnse_dict['UNet2'] = {}
    mnse_dict['ResResNet'] = {}
    # mnse_dict['ResNet'] = {}

    for wrn_mnse_line in wrn_mnse_lines:
        split_values = wrn_mnse_line.split('"')
        if split_values[0] != '':
            start_index = split_values[0].find('DL-') + len('DL-')
            end_index = split_values[0].find('_DBT')
            dl_model = split_values[0][start_index:end_index]

            start_index = split_values[0].find('rnw') + len('rnw')
            end_index = split_values[0].find('-30mAs')
            rnw = float(split_values[0][start_index:end_index])

            # mnse = split_values[1]
            rn = split_values[3]
            bias = split_values[5]

            mnse_dict[dl_model][rnw] = {'bias': float(bias.split(' ')[0]),
                                        'rn': float(rn.split(' ')[0])}

    return mnse_dict


# %%

path2read = ".."
path2write = '/home/rodrigo/Dropbox/'

# %%


mnse_dict = extract_bias_rn("{}/tmp/outputs".format(path2read))

names2print = ['RED-CNN', 'U-Net', 'ResNet']
colors2print = ['red', 'green', 'black']

# %% Plotando com epoch em fig diferente

fig2, ax3 = plt.subplots()

lns_fd, = ax3.plot(1000*0.0009147, 0.1186190, "P", c='black', markersize=8, markerfacecolor='black', label='FD')
lns_mb, = ax3.plot(1000*0.0026481, 0.1323195, "*", c='black', markersize=8, markerfacecolor='none', label='MB')

lns_hbl, = ax3.plot(1000*0.0033301, 0.1419810, "D", c=colors2print[2], markersize=8, markeredgewidth=2, markerfacecolor='none', label='HBL') # ResNet
_, = ax3.plot(1000*0.0030587, 0.1415164, "D", c=colors2print[1], markersize=8, markeredgewidth=2, markerfacecolor='none', label='Unet')
_, = ax3.plot(1000*0.0027211, 0.1414709, "D", c=colors2print[0], markersize=8, markeredgewidth=2, markerfacecolor='none', label='RED-CNN')

lns_dbl, = ax3.plot(1000*0.0082829, 0.1111480, "o", c=colors2print[2], markersize=8, markeredgewidth=2, markerfacecolor='none', label='DBL') # ResNet
_, = ax3.plot(1000*0.0070128, 0.1123836, "o", c=colors2print[1], markersize=8, markeredgewidth=2, markerfacecolor='none', label='Unet')
_, = ax3.plot(1000*0.0042103, 0.1040086, "o", c=colors2print[0], markersize=8, markeredgewidth=2, markerfacecolor='none', label='RED')


for color2print, name2print, model_key in zip(colors2print, names2print, mnse_dict):
    wrn_all = np.array(list(mnse_dict[model_key].keys()))
    bias_ep = np.array([mnse_dict[model_key][key]['bias'] for key in mnse_dict[model_key]])
    rn_ep = np.array([mnse_dict[model_key][key]['rn'] for key in mnse_dict[model_key]])

    threshold = 0.05
    up_bound = 0.1186 * (1 + threshold)
    low_bound = 0.1186 * (1 - threshold)

    inds = (rn_ep < up_bound) & (rn_ep > low_bound)
    selected_wrns = wrn_all[inds]
    selected_bias = bias_ep[inds]
    selected_rn = rn_ep[inds]

    ind_lowest_bias = np.argmin(selected_bias)

    lns_cl, = ax3.plot(1000*selected_bias[ind_lowest_bias], selected_rn[ind_lowest_bias], "s", markersize=8, markeredgewidth=2, markerfacecolor='none', c=color2print, label='$\lambda_{{RN}}$')

    fig, ax1 = plt.subplots()

    # Plot selected point
    lns4, = ax1.plot(np.array((selected_wrns[ind_lowest_bias], selected_wrns[ind_lowest_bias])),
                     np.array((1.5, 4.5)),
                     '--',
                     color='green',
                     alpha=.6,
                     linewidth=2,
                     label=r'Selected point: $\lambda_{{RN}}$:{:.3f} $B^2$={:.2f}, RN={:.4f}'.format(
                         selected_wrns[ind_lowest_bias],
                         1000 * selected_bias[ind_lowest_bias],
                         selected_rn[ind_lowest_bias]))

    # Plot bias
    ax1.set_ylabel('$B^2 (10^{-3})$', fontsize=14)
    ax1.set_xlabel('$\lambda_{RN}$', fontsize=14)
    lns1, = ax1.plot(wrn_all, 1000 * bias_ep, '.', color='red', label='$B^2$')
    ax1.set_ylim(1.5, 4.5)
    ax1.set_xlim(0.0, 0.5)
    ax1.tick_params(axis='both', labelsize=14)

    # Plot RN
    ax2 = ax1.twinx()
    ax2.set_ylabel('RN', fontsize=14)
    ax2.set_xlabel("$\lambda_{RN}$", fontsize=14)
    lns3, = ax2.plot(np.linspace(np.min(wrn_all), np.max(wrn_all), num=wrn_all.shape[0]),
                     np.ones_like(wrn_all) * 0.1186, '--', color='lightblue', label='Full-dose RN', linewidth=2)
    lns2, = ax2.plot(wrn_all, rn_ep, '.', color='blue', label='RN')
    ax2.set_ylim(0.10, 0.24)
    ax2.set_xlim(0.0, 0.5)
    ax2.tick_params(axis='both', labelsize=14)

    # for k in range(wrn_all.shape[0]):
    #     ax1.plot(np.array((wrn_all[k], wrn_all[k])),
    #               np.array((0.0023, 0.0040)),
    #               '--',
    #               color='black',
    #               alpha=.2)

    ax1.grid()
    ax1.legend(handles=[lns1, lns2, lns3, lns4], prop={'size': 9}, loc='upper center')
    ax1.set_title(name2print, fontsize=18)

    fig.tight_layout()
    fig.savefig(path2write + model_key + "_wRNxb2xRN_ep02.png", bbox_inches="tight")

    print(r'{}: Wrn:{} B^2={}, RN={}'.format(model_key,
                                             selected_wrns[ind_lowest_bias],
                                             selected_bias[ind_lowest_bias],
                                             selected_rn[ind_lowest_bias]))

# (0.16, 0)(0, 0.16)
#
# (0.15, 0)(0, 0.15)
#
# (0.14, 0)(0, 0.14)
#
# (0.13, 0)(0, 0.13)
#
# (0.12, 0)(0, 0.12)
#
# (0.11, 0)(0, 0.11)
#
# (0.1, 0)(0, 0.1)

_, = ax3.plot([100, 0], [0, 0.1], '-.', color='black', linewidth=2, alpha=.1,)
offset = 0.01
for k in range(1, 7):
    _, = ax3.plot([100 + (1000*k*offset), 0], [0, 0.1 + (k*offset)], '-.', color='black', linewidth=2, alpha=.1,)
    # print([100 + (1000*k*offset), 0], [0, 0.1 + (k*offset)])

# ax3.grid()
ax3.set_ylim(0.1, 0.15)
ax3.set_xlim(0, 9)
ax3.set_xlabel('$B^2 (10^{-3})$', fontsize=14)
ax3.set_ylabel('RN', fontsize=14)
ax3.tick_params(axis='both', labelsize=14)
ax3.legend(handles=[lns_fd, lns_mb, lns_hbl, lns_dbl, lns_cl], prop={'size': 14}, loc='upper right')

fig2.tight_layout()
fig2.savefig(path2write + "b2xRN.png", bbox_inches="tight")



