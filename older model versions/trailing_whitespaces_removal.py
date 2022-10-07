# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 09:47:01 2022

@author: earyo
"""

with open("country_model_2_data_version.py") as file:
    for line in file:
        line = line.rstrip()
        if line:
            print(line)