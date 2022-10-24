# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:06:38 2022

@author: earyo
"""

#%% Loading libraries

### import necessary libraries
import os
import mesa
import mesa.time
import mesa.space
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math as math
import pandas as pd
import copy as copy
from math import radians, cos, sin, asin, sqrt
import random
from datetime import datetime as dt
import sys
# import random 
from random import sample
### colormaps import
import matplotlib.cm
##
from multiprocessing import Pool
##
import pytest
### import model class (which itself imports agent class)
from model_class2 import CountryModel


#%% RUN THE MODEL
### run the model, without particle filter, and save data

def run_base_model():
    
            ### call the model iteration
            ##4th parameter initial conditions can be real, no countries yet or random
            model = CountryModel(0.01, 0.13, 18, 'real', 'no')
            for i in range(31):
                            
                            model.step()
                            df1 = model.datacollector.get_agent_vars_dataframe()
                            df2 = model.datacollector.get_model_vars_dataframe()
                                    
            ### here insert code for merging dataframes
            df3 = (df1.reset_index(level=0)).reset_index(level = 0)
            df4 = pd.DataFrame(np.repeat(df2.values, 200, axis=0))
            df4.columns = df2.columns
            df4 = pd.concat([df4, df3], axis=1, join='inner')
            df4 = df4[["code","AgentID", "Step", "minimum_difference", "Lockdown", "income",
                       "politicalregime", "social_thre", "own_thre",
                       "adoption_mode"]]
            #df4.insert(0, "iteration", [j]*len(df4))
            
            return df4