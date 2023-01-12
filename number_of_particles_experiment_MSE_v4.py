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
from multiprocessing import Pool
from itertools import starmap
import csv

#work laptop path
os.chdir("C:/Users/earyo/Dropbox/Arbeit/postdoc_leeds/ABM_python_first_steps/implement_own_covid_policy_model")
##
import pytest
### import model class (which itself imports agent class)
from model_class2 import CountryModel
### import NON-parallelized particle filter (because here the experiment itself gets parallelized 
## and one cannot have parallelization within parallelization)
from particle_filter_class import ParticleFilter
# import random 
from run_base_model import run_base_model
from experiment import Experiment


if __name__ == '__main__':
    #READ DATA
    ### read country/agent data
    with open('C:/Users/earyo/Dropbox/Arbeit/postdoc_leeds/ABM_python_first_steps/implement_own_covid_policy_model/data/agent_data_v2.csv') as f:
        agent_data = pd.read_csv(f, encoding = 'unicode_escape')
        
    Num_agents = len(agent_data)
    agent_data["gdp_pc"] = pd.to_numeric(agent_data["gdp_pc"])
    
    ##### Read data for calibration
    #### aggregate diffusion curve data
    with open('C:/Users/earyo/Dropbox/Arbeit/postdoc_leeds/ABM_python_first_steps/implement_own_covid_policy_model/data/lockdown_diffusion_curve_updated_for_calibration.csv') as f:
        lockdown_data1 = pd.read_csv(f, encoding = 'unicode_escape', header = None)
    
    #### data per country
    with open('C:/Users/earyo/Dropbox/Arbeit/postdoc_leeds/ABM_python_first_steps/implement_own_covid_policy_model/data/lockdown_tracking.csv') as f:
        lockdown_data2  = pd.read_csv(f, encoding = 'unicode_escape')

    #%%    
    
    #print(run_experiment(num_power_particles))
    #print(str(__name__ == '__main__') + "xxxx")
    #num_power_particles_list = [2,4,8,16,32,64,128,256,512,1024]
    
    num_power_particles_list = [1]
    iterations_filter = 1
    
    #### this must be multiplied by the len of the number of particles as well as the iterations
    ### per particle number
    lockdown_data1_list = [lockdown_data1]*len(num_power_particles_list)*iterations_filter
    lockdown_data2_list = [lockdown_data2]*len(num_power_particles_list)*iterations_filter
    
    
    number_of_particles_per_experiment = [2**x for x in num_power_particles_list]
    results_mse_all = []
    results_msepf_all = []
    results_micro_all = []
    results_micro_all_pf = []
    
    if iterations_filter <= 8:
        number_of_processors = iterations_filter
    else:
        number_of_processors = 8
    
    iterations_as_columns = np.linspace(1,iterations_filter,iterations_filter)
    start = datetime.now()
    
    start = datetime.now()
    for num_power_particles in num_power_particles_list:
        
        # Create a multiprocessing Pool
        ### use itertools.starmap instead of pool.starmap
        ### make sure to debug
   
        pool_input = [num_power_particles]*iterations_filter
        #print(pool_input)
        #### inputs_for_starmap must be sorted so that first all runs with x particles are done
        #### and subsequently only with y particles, where x < y etc. 
        ### it needs to be done like this sorted(list(zip([0,1]*7,[8]*7*2,[9]*7*2)))
        
        inputs_for_starmap = sorted(list(zip(pool_input, lockdown_data1_list, lockdown_data2_list)))
        #print(inputs_for_starmap)
        #print(np.array(list(inputs_for_starmap)).shape)    
        with Pool(processes=number_of_processors) as pool:     
            data_results = list(pool.starmap(Experiment.run_experiment, inputs_for_starmap))
       # assert len(data_results) == len(num_power_particles_list), f"The length of the results isn't what we expected {len(num_power_particles_list)}"
        #print("this is data results", data_results)
        results_mse =[]
        results_msepf =[]
        results_micro = []
        results_micro_pf = []
        for i in range(iterations_filter):
                ### here the sum over time is taken of the MSE. so it is basically
                ### an approximation to the integral of the MSE over time. since
                ### one time step equals one unit and every rectangle then is 1*MSE,
                ### the sum suffices for simple numerical integral. The smaller the area,
                ### the better the particles work 
                results_mse.append(sum(data_results[i][0]))
                results_msepf.append(sum(data_results[i][1])) 
                results_micro.append(data_results[i][2])
                results_micro_pf.append(data_results[i][3])
        
        results_mse_all.append(results_mse)
        results_msepf_all.append(results_msepf)
        results_micro_all.append(results_micro)
        results_micro_all_pf.append(results_micro_pf)
        print(f"Completed the iterations with  {2**num_power_particles} particles")
    
    
    #C:/Users/earyo/Dropbox/Arbeit/postdoc_leeds/ABM_python_first_steps/implement_own_covid_policy_model/
    df1 = pd.DataFrame(results_mse_all, columns = iterations_as_columns)
    df2 = pd.DataFrame(results_msepf_all, columns = iterations_as_columns)
    df3 = pd.DataFrame(results_micro_all, columns = iterations_as_columns)
    df4 = pd.DataFrame(results_micro_all_pf, columns = iterations_as_columns)
    #df.set_index(pd.Series(number_of_particles_per_experiment))
    df1.index.name = 'Number of particles'
    df1.to_csv("N_of_particles_exp_without_pf.csv", sep=',')
    #df.set_index(pd.Series(number_of_particles_per_experiment))
    df2.index.name = 'Number of particles'
    df2.to_csv("N_of_particles_exp_with_pf.csv", sep=',')
    df3.to_csv("micro_validity_without_pf.csv", sep=',')
    df4.to_csv("micro_validity_with_pf.csv", sep=',')
    
    
    finish = datetime.now()
    print("time to execute was: ", (finish - start).total_seconds())
    
    ##### tidy up the code a bit, put it into classes
    
    ### make sure the effort/overhead to put up parallel processing is worth it 
    ### https://stackoverflow.com/questions/20727375/multiprocessing-pool-slower-than-just-using-ordinary-functions



#%%

