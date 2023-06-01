#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:28:51 2022

@author: Rodrigo
"""

import os

from libs.utilities import removedir, makedir

python_path = "/home/rodrigo/Documents/Rodrigo/Python_VE/science/bin/python"
# python_path = "/home/laviusp/Documents/pyvenv/science/bin/python"

restoration_path = '/media/rodrigo/Dados_2TB/Imagens/UPenn/Phantom/Anthropomorphic/DBT/Restorations/31_30/'
# restoration_path = '/home/laviusp/Documents/Rodrigo_Vimieiro/phantom/Restorations/31_30'

rnw_weights = [0.00000000, 0.10526316, 0.21052632, 0.31578947, 0.42105263,
               0.52631579, 0.63157895, 0.73684211, 0.84210526, 0.94736842,
               1.05263158, 1.15789474, 1.26315789, 1.36842105, 1.47368421,
               1.57894737, 1.68421053, 1.78947368, 1.89473684, 2.00000000,
               0.00501253, 0.01002506, 0.01503759, 0.02005013, 0.02506266,
               0.03007519, 0.03508772, 0.04010025, 0.04511278, 0.05012531,
               0.05513785, 0.06015038, 0.06516291, 0.07017544, 0.07518797,
               0.08020050, 0.08521303, 0.09022557, 0.09523810, 0.10025063]

# rnw_weights = [0.0031493]

models = ['ResResNet', 'UNet2', 'RED']

# models = ['RED']

framework = 'Noise2Sim'#'VSTasLayer-MNSE'  #
n_epochs = 5

for model in models:
    for rnw_weight in rnw_weights:

        ##########  Train ##########
        python_command = "{} main_training_MNSE.py \
                  --nep {} \
                  --rnw {} \
                  --model {}".format(python_path,
                                     n_epochs,
                                     rnw_weight,
                                     model)

        print(python_command)
        os.system(python_command)

        #########  Test ##########
        python_command = "{} main_testing.py \
                          --nep {} \
                          --rnw {} \
                          --model {} \
                          --fmw {}".format(python_path,
                                             n_epochs,
                                             rnw_weight,
                                             model,
                                             framework)

        print(python_command)
        makedir(restoration_path)
        os.system(python_command)

        ##########  MNSE ##########
        python_command = "{} evaluation/MNSE.py \
                  --model {}_DBT_{}_rnw{} --rf {:d}".format(python_path,
                                                            model,
                                                            framework,
                                                            rnw_weight,
                                                            50)

        print(python_command)
        os.system(python_command)

        ##########  SNR ##########
        python_command = "{} evaluation/SIIM_QILV_SNR.py \
                          --model {}_DBT_{}_rnw{} --rf {:d}".format(python_path,
                                                                    model,
                                                                    framework,
                                                                    rnw_weight,
                                                                    50)

        print(python_command)
        os.system(python_command)



        # removedir(restoration_path)
