# -*- coding: utf-8 -*-
"""
Covid lockdown adoption model script and particle filter
modified: 08/09/2022
modified and created by: Yannick Oswald
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
from datetime import datetime as datetime
import sys
# import random 
from random import sample
### colormaps import
import matplotlib.cm
##
import multiprocessing as mp
from multiprocessing import Pool
import csv

#work laptop path
os.chdir("C:/Users/earyo/Dropbox/Arbeit/postdoc_leeds/ABM_python_first_steps/implement_own_covid_policy_model")

##
import pytest
### import model class (which itself imports agent class)
from model_class2 import CountryModel
from particle_filter_class import ParticleFilter
# import random 
from run_base_model import run_base_model

#%% READ DATA
### read country/agent data
agent_data = pd.read_csv('agent_data_v2.csv', encoding = 'unicode_escape')
Num_agents = len(agent_data)
agent_data["gdp_pc"] = pd.to_numeric(agent_data["gdp_pc"])

##### Read data for calibration
#### aggregate diffusion curve data
lockdown_data1 = pd.read_csv('lockdown_diffusion_curve_updated_for_calibration.csv', 
                             encoding = 'unicode_escape',
                             header = None)
#### data per country
lockdown_data2 = pd.read_csv('lockdown_tracking.csv', 
                             encoding = 'unicode_escape')      
#%% 


### https://stackoverflow.com/questions/20190668/multiprocessing-a-for-loop

#mse_array = np.zeros(iterations_filter, num_power_particles)
#mse_array_pf = np.zeros(iterations_filter, num_power_particles)



def run_experiment(num_power):
        ### goes to num_power - 1
        #RUN THE MODEL
        ### run the model, without particle filter, and save data
        #print("am i doing sth.?")
        testy = run_base_model(2**num_power)
        array_run_results = testy[1]
        
        # RUN PARTICLE FILTER EXPERIMENTS 
        
        pf_parameters = {
          "da_window": 5,
          "da_instances": 30/5,
          "No_of_particles": 2**num_power
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
        
        
        ##### PLOT mean squared error per time step. pf results vs no pf results        
        
        results_pf_percent = results_pf/164
        results_percent = array_run_results.T/164
        square_diffs_pf = np.zeros((31,No_of_particles))
        square_diffs = np.zeros((31,No_of_particles))
        for i in range(No_of_particles):
            square_diffs_pf[:,i] = (results_pf_percent[:,i] - lockdown_data1.iloc[:,0].to_numpy())**2
            square_diffs[:,i] = (results_percent[:,i] - lockdown_data1.iloc[:,0].to_numpy())**2 
        
        mse_pf =  np.mean(square_diffs_pf, axis = 1)
        mse =  np.mean(square_diffs, axis = 1)
        
        return mse, mse_pf
    
    
#%%    

#print(run_experiment(num_power_particles))
#print(str(__name__ == '__main__') + "xxxx")


#num_power_particles_list = [2,4,8,16,32,64,128,256,512,1024]

num_power_particles_list = [2,3,4,5,6]
number_of_particles_per_experiment = [2**x for x in num_power_particles_list]
results_mse_all = []
results_msepf_all = []

iterations_filter = 20
iterations_as_columns = np.linspace(1,iterations_filter,iterations_filter)
start = datetime.now()

start = datetime.now()
for num_power_particles in num_power_particles_list:
    
    if __name__ == '__main__':
        # Create a multiprocessing Pool
        pool = Pool(processes = 8)  
        pool_input = list(np.repeat(num_power_particles, iterations_filter))                      
        data_results = pool.map(run_experiment, pool_input)
        results_mse =[]
        results_msepf =[]
        for i in range(iterations_filter):
                ### here the sum over time is taken of the MSE. so it is basically
                ### an approximation to the integral of the MSE over time. since
                ### one time step equals one unit and every rectangle then is 1*MSE,
                ### the sum suffices for simple numerical integral. The smaller the area,
                ### the better the particles work 
                results_mse.append(sum(data_results[i][0]))
                results_msepf.append(sum(data_results[i][1])) 
        
        results_mse_all.append(results_mse)
        results_msepf_all.append(results_msepf)
        print("Completed the iterations with ", 2**num_power_particles, "particles")
        
df1 = pd.DataFrame(results_mse_all, columns=iterations_as_columns)
#df.set_index(pd.Series(number_of_particles_per_experiment))
df1.index.name = 'Number of particles'
df1.to_csv("N_of_particles_exp_without_pf.csv", sep=',')
df2 = pd.DataFrame(results_msepf_all, columns=iterations_as_columns)
#df.set_index(pd.Series(number_of_particles_per_experiment))
df2.index.name = 'Number of particles'
df2.to_csv("N_of_particles_exp_with_pf.csv", sep=',')


finish = datetime.now()
print("time to execute was: ", (finish - start).total_seconds())

##### tidy up the code a bit, put it into classes    
    
#%%    
#print(result)





                      #%%
                      
                      #mse_array = mse
                      #mse_array_pf = mse_pf
                      
                      #plt.scatter(2**num_power,sum(mse), label = "no filter", color = "tab:blue")
                      #plt.scatter(2**num_power,sum(mse_pf), label = "particle filter", color = "tab:red")
                      #plt.xscale("log", base=2)
                      #plt.xlabel("Number of particles considered")
                      #plt.ylabel("Sum of MSEs over time")  
                      #print(f"This iteration is {2**num_power} number of particles, tested the {itr} time")
                  
              ##this super nice code here
              ##https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend        
              #handles, labels = plt.gca().get_legend_handles_labels()
              #by_label = dict(zip(labels, handles))
              #plt.legend(by_label.values(), by_label.keys(), frameon = False, loc=(1.05, 0.5))