# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:51:55 2022

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
from model_class import CountryModel

#### data per country
with open('C:/Users/earyo/Dropbox/Arbeit/postdoc_leeds/ABM_python_first_steps/implement_own_covid_policy_model/data/lockdown_tracking.csv') as f:
    lockdown_data2  = pd.read_csv(f, encoding = 'unicode_escape')


#%% RUN THE MODEL
### run the model, without particle filter, and save data

def run_base_model(no_of_iterations):
    
    for j in range(no_of_iterations):
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
            df4.insert(0, "iteration", [j]*len(df4))
            
            if j == 0 :
                df_results = pd.DataFrame(columns = df4.columns)
                df_results = pd.concat([df_results, df4], join="inner")
            else:
                df_results = pd.concat([df_results, df4], join="inner", ignore_index=True)
            
            #print("model iteration is " + str(j))
    
    array_run_results = np.zeros((no_of_iterations,31))
    micro_validity_metric_array = np.zeros((31,no_of_iterations))
    for i in range(no_of_iterations):
        array_run_results[i,:] = np.sum(np.split(np.array(df_results[(df_results.iteration == i)]["Lockdown"]),31),axis=1)
        alpha = pd.Series.reset_index(df_results[(df_results.iteration == i) ]["Lockdown"], drop = True) 
        beta =  pd.Series.reset_index(lockdown_data2.sort_values(['model_step', 'Entity'])["lockdown"],drop = True)          
        micro_validity_metric_array[:,i] = np.mean(np.array_split(np.array(alpha == beta),31),axis = 1)
    
    return df_results, array_run_results, micro_validity_metric_array


