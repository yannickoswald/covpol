# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 13:30:42 2022

@author: earyo
"""



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
from particle_filter_class import ParticleFilter
from run_base_model_opt import model_run
from multiprocessing import Pool

#work laptop path
os.chdir("C:/Users/earyo/Dropbox/Arbeit/postdoc_leeds/ABM_python_first_steps/implement_own_covid_policy_model")



#%% RUN PARTICLE FILTER EXPERIMENTS
 

pf_parameters = {
  "da_window": 5,
  "da_instances": 30/5,
  "No_of_particles": 10
}


model_parameters = {
  "base_alert": 0.01,
  "social_base_threshold": 0.13,
  "clique_size": 18,
  "initial_conditions": 'real',
  "data_update": 'no'
}

current_PF = ParticleFilter(CountryModel, model_parameters, pf_parameters)  
current_PF.run_particle_filter()


#%% PLOTTING PARTICLE FILTER RESULTS 

 ### TO DO verify whether pf works correctly 
No_of_particles = pf_parameters['No_of_particles']
da_window = pf_parameters['No_of_particles']
results_pf = np.zeros((31, No_of_particles))
micro_validity_metric_array_pf = np.zeros((31, No_of_particles))

Dit={}
time_steps = 31
for i in range(31):
    for j in range(No_of_particles):
        ### key is a tuple where i equals time step, j particle number and
        ### the third number the model id, initially unique, but later can
        ### be associated with several particles because they are resampled 
        ### versions of one another. Therefore ultimately j is the *unique* 
        ### identifier as well as the tuple as a whole given j. 
        key = (i,j, current_PF.part_filtered_all[i][j].model_id)
        value = current_PF.part_filtered_all[i][j].datacollector.get_agent_vars_dataframe()
        Dit[key] = value
        df = Dit[key]
        df = (df.reset_index(level=0)).reset_index(level = 0)
        results_pf[i,j] = df[(df.Step == i)]["Lockdown"].sum()
        micro_state = pd.Series.reset_index(df[(df.Step == i)]["Lockdown"],drop = True)
        micro_state_data = pd.Series.reset_index(lockdown_data2[(lockdown_data2.model_step == i)]["lockdown"],drop = True) 
        micro_validity_metric_array_pf[i,j] = np.mean(micro_state == micro_state_data)
        print(i,j)
        



##### PLOT mean squared error per time step. pf results vs no pf results 
results_pf_percent = results_pf/164
square_diffs_pf = np.zeros((31,No_of_particles))
for i in range(No_of_particles):
    square_diffs_pf[:,i] = (results_pf_percent[:,i] - lockdown_data1.iloc[:,0].to_numpy())**2

mse_pf =  np.mean(square_diffs_pf, axis = 1)

mse_pf =  np.mean(square_diffs_pf, axis = 1)
mse =  np.array(mse_as_list)


plt.plot(np.linspace(1,31,31), mse)
plt.plot(np.linspace(1,31,31), mse_pf)
plt.xlabel("Day of March")
plt.ylabel("Mean squared error") ### perhaps plot squared error as fan-chart around?
plt.savefig('MSE_over_time.png', bbox_inches='tight', dpi=300)
plt.show()


mse_list = [sum(mse), sum(mse_pf)]
df_mse = pd.DataFrame(mse_list)

df_mse.to_csv("df_mse.csv", sep=',')


#%%

def create_fanchart_2_PF(arr):
    x = np.arange(arr.shape[0])+1
    # for the median use `np.median` and change the legend below
    mean = np.mean(arr, axis=1)
    offsets = (25,67/2,47.5)
    fig, ax = plt.subplots()
    ax.plot(x, mean, color='black', lw=3)
    for offset in offsets:
        low = np.percentile(arr, 50-offset, axis=1)
        high = np.percentile(arr, 50+offset, axis=1)
        # since `offset` will never be bigger than 50, do 55-offset so that
        # even for the whole range of the graph the fanchart is visible
        alpha = (55 - offset) / 100
        ax.fill_between(x, low, high, color='tab:red', alpha=alpha)
        ax.set_xlabel("Day of March")
        ax.set_ylabel("% of countries in correct state")
        ax.legend(['Mean'] + [f'Pct{int(2*o)}' for o in offsets] + ['data'], frameon = False)
    ax.margins(x=0)
    return fig, ax


create_fanchart_2_PF(micro_validity_metric_array_pf*100)
plt.savefig('fanchart_2_micro_validity_pf.png', bbox_inches='tight', dpi=300)
plt.show()








#%% a few more data visuals for exploration of the model and sup mat.



"""

##### microvalidity pf vs. no pf as a chart of the mean lines

fig3, ax1 = plt.subplots(figsize = (5.5,5))
arr1 = micro_validity_metric_array*100
arr2 = micro_validity_metric_array_pf*100
x = np.arange(arr1.shape[0])+1
y = np.arange(arr2.shape[0])+1
# for the median use `np.median` and change the legend below
mean1 = np.mean(arr1, axis=1)
mean2 = np.mean(arr2, axis=1)
offsets = (25,67/2,47.5)
ax1.plot(x, mean1, color='black', lw=3, label = "no_pf")
ax1.plot(y, mean2, color='tab:red', lw=3, label = "with_pf")
ax1.set_xlabel("Day of March")
ax1.set_ylabel("% in correct state")
ax1.legend()
"""






#%%
"""


[x.social_thre for x in model.schedule.agents]



testdistr = [x.own_thre for x in model.schedule.agents]

plt.hist(testdistr, density=True, bins=30)
plt.xlabel("initiative threshold")
plt.ylabel("number of countries")
plt.show


testdistr2 = [x.social_thre for x in model.schedule.agents]

plt.hist(testdistr2, density=True, bins=30)
plt.xlabel("social threshold")
plt.ylabel("number of countries")

"""