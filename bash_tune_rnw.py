#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:28:51 2022

@author: Rodrigo
"""

import os

python_path = "/home/rodrigo/.virtualenvs/science/bin/python"

reduc_factors = [50]
rnw_weights = ['0.01357164'] #['0.1357164','3.3800761', '9.4957105'] # ['']#

model_type = 'VSTasLayer-MNSE_rnw' #'Noise2Sim'#

## now loop through the above array
for reduc_factor in reduc_factors:
    for rnw_weight in rnw_weights:
                
        # ##########  Train ##########
        python_command = "{} main_training_MNSE.py \
                  --rf {} \
                  --rfton {} \
                  --rnw {}".format(python_path, reduc_factor, 50, rnw_weight)

        print(python_command)
        os.system(python_command)
        
        ##########  Test ##########
        python_command = "{} main_testing.py \
                  --rf {} \
                  --rfton {} \
                  --model {}{}".format(python_path, reduc_factor, 50, model_type, rnw_weight)

        print(python_command)
        os.system(python_command)
        
        ##########  MNSE ##########
        
        python_command = "{} evaluation/MNSE.py \
                  --rf {} \
                  --model {}{}".format(python_path, reduc_factor, model_type, rnw_weight)

        print(python_command)
        os.system(python_command)
        
        
        
        
        
    

