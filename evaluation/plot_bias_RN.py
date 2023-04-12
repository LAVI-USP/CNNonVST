#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 16:27:26 2022

@author: rodrigo
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

#%%

def extract_bias_rn(file_name):
    
    f = open(file_name + ".txt", "r")
    wrn_mnse_lines = f.read().split('\n')
    
    bias_partex = []
    rn_partex = []
    
    for wrn_mnse_line in wrn_mnse_lines:
        split_values = wrn_mnse_line.split('"') 
        if split_values[0] != '':
            # mnse = split_values[1]
            rn = split_values[3]
            bias = split_values[5]
            
            bias_partex.append(float(bias.split(' ')[0]))
            rn_partex.append(float(rn.split(' ')[0]))
            
    f.close()
      
    return bias_partex, rn_partex

#%%

path2read = "."

wrn_all = [0.11436859, 0.4746975, 0.6235101, 0.33800761, 0.67475232, 0.31720174, 0.77834548, 0.94957105, 0.66252687, 0.01357164,
              0.00636544, 0.02196585, 0.03210443, 0.03833522, 0.04172908, 0.05781851, 0.06680424, 0.06866511, 0.09055672, 0.10951304, 0 ,1]

#%%


bias_ep02, rn_ep02 = extract_bias_rn("{}/RED".format(path2read))


#%% Plotando com epoch em fig diferente

# fig, ax1 = plt.subplots()

# ax1.set_ylabel('$B^2$ (%)')
# ax1.set_xlabel('$\lambda_{RN}$')
# lns1, = ax1.plot(wrn_all, bias_ep01, '.', color='red', label=r'$\lambda_{{RN}}$:0 $B^2$={}% | $\lambda_{{RN}}$:$\infty$ $B^2$={}%'.format(bias_ep01_[0], bias_ep01_[1]))

# ax2 = ax1.twinx()

# ax2.set_ylabel('Residual Noise (%)')
# ax2.set_xlabel('$\lambda_{RN}$')
# lns2, = ax2.plot(wrn_all, rn_ep01, '.', color='blue', label=r'$\lambda_{{RN}}$:0 RN={}% | $\lambda_{{RN}}$:$\infty$ RN={}%'.format(rn_ep01_[0], rn_ep01_[1]))

# ax1.grid()
# ax1.legend(handles=[lns1, lns2], prop={'size': 9}, loc='upper center')
# ax1.set_title("Epoch 1")

# fig.tight_layout()  
# fig.savefig("wRNxb2xRN_ep01.png", bbox_inches="tight")

wrn_all = np.array(wrn_all)

fig, ax1 = plt.subplots()

ax1.set_ylabel('$B^2$ (%)')
ax1.set_xlabel('$\lambda_{RN}$')
lns1, = ax1.plot(wrn_all[:-2], bias_ep02[:-2], '.', color='red', label=r'$\lambda_{{RN}}$:0 $B^2$={}% | $\lambda_{{RN}}$:$\infty$ $B^2$={}%'.format(bias_ep02[-2], bias_ep02[-1]))


ax2 = ax1.twinx()

ax2.set_ylabel('Residual Noise (%)')
ax2.set_xlabel("$\lambda_{RN}$")
lns2, = ax2.plot(wrn_all[:-2], rn_ep02[:-2], '.', color='blue', label=r'$\lambda_{{RN}}$:0 RN={}% | $\lambda_{{RN}}$:$\infty$ RN={}%'.format(rn_ep02[-2], rn_ep02[-1]))
lns3, = ax2.plot(np.linspace(np.min(wrn_all[:-2]), np.max(wrn_all[:-2]), num=wrn_all[:-2].shape[0]), np.ones_like(wrn_all[:-2]) * 11.86, '--', color='green', label='Full-dose RN')


ax1.grid()
ax1.legend(handles=[lns1, lns2, lns3], prop={'size': 9}, loc='center')
ax1.set_title("Epoch 2")

fig.tight_layout()  
fig.savefig("wRNxb2xRN_ep02.png", bbox_inches="tight")


#%% Plotando na mesma figura 

# fig, ax1 = plt.subplots(figsize=(5, 5))

# ax1.set_ylabel('$B^2$ (%)"', color='tab:red')
# ax1.set_xlabel("$\lambda_{RN}$")
# lns11, = ax1.plot(wrn_all, bias_ep01, '.', markersize=10, linewidth=1.5, color='red', label=r'Ep1 - wRN:0 $B^2$={}% | wRN:$\infty$ $B^2$={}%'.format(bias_ep01_[0], bias_ep01_[1]))
# lns12, = ax1.plot(wrn_all, bias_ep02, 'x', markersize=10, linewidth=1.5, color='red', label=r'Ep2 - wRN:0 $B^2$={}% | wRN:$\infty$ $B^2$={}%'.format(bias_ep02_[0], bias_ep02_[1]))
# # ax1.tick_params(axis='y', labelcolor='tab:red')
# # ax1.yaxis.set_major_formatter(money_formatter)
# ax1.grid(True)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# ax2.set_ylabel('Residual Noise (%)', color='tab:blue')
# ax2.set_xlabel("$\lambda_{RN}$")
# lns21, = ax2.plot(wrn_all, rn_ep01, '.', markersize=10, linewidth=1.5, color='blue', label=r'Ep1 - wRN:0 RN={}% | wRN:$\infty$ RN={}%'.format(rn_ep01_[0], rn_ep01_[1]))
# lns22, = ax2.plot(wrn_all, rn_ep02, 'x', markersize=10, linewidth=1.5, color='blue', label=r'Ep2 - wRN:0 RN={}% | wRN:$\infty$ RN={}%'.format(rn_ep02_[0], rn_ep02_[1]))

# ax1.legend(handles=[lns11, lns12, lns21, lns22])

# fig.tight_layout()  # otherwise the right y-label is slightly clipped













# Make data.
# X, Y = np.meshgrid(wrn_parte1, bias_parte1_ep01)
# Z = np.array(rn_parte1_ep01)
# # Z = np.vstack((bias_parte1_ep01, bias_parte1_ep02))

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                         linewidth=0, antialiased=False)


# ax.set_xlabel('WRN')
# ax.set_ylabel('Epochs')
# ax.set_zlabel('Bias')






