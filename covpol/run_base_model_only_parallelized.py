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
from particle_filter_class import ParticleFilter
from run_base_model_opt import model_run
from multiprocessing import Pool
import multiprocessing

#work laptop path
#os.chdir("C:/Users/earyo/Dropbox/Arbeit/postdoc_leeds/ABM_python_first_steps/implement_own_covid_policy_model")


if __name__ == '__main__':

    #%% READ DATA
    ### read country/agent data
    with open('../data/agent_data_v2.csv') as f:
        agent_data = pd.read_csv(f, encoding='unicode_escape')

    Num_agents = len(agent_data)
    agent_data["gdp_pc"] = pd.to_numeric(agent_data["gdp_pc"])

    ##### Read data for calibration
    #### aggregate diffusion curve data
    with open('../data/lockdown_diffusion_curve_updated_for_calibration.csv') as f:
        lockdown_data1 = pd.read_csv(f, encoding='unicode_escape', header=None)

    #### data per country
    with open('../data/lockdown_tracking.csv') as f:
        lockdown_data2 = pd.read_csv(f, encoding='unicode_escape')

    start = dt.now()

    # Create a multiprocessing Pool
    
    number_of_processors = multiprocessing.cpu_count()
    number_of_runs = 100
    lockdown_data1_list = [lockdown_data1]*number_of_runs
    lockdown_data2_list = [lockdown_data2]*number_of_runs

    inputs_for_starmap = list(zip(lockdown_data1_list, lockdown_data2_list))
    #print(inputs_for_starmap)
    #print(np.array(list(inputs_for_starmap)).shape)
    with Pool(processes=number_of_processors) as pool:
        data_results = list(pool.starmap(
            model_run.run_base_model_opt, inputs_for_starmap))
    # assert len(data_results) == len(num_power_particles_list), f"The length of the results isn't what we expected {len(num_power_particles_list)}"
    #print("this is data results", data_results)

    running_secs = (dt.now() - start).seconds
    print("running time was " + str(running_secs) + " sec")

    square_error_list = [x[1] for x in data_results]
    mse_as_list = [sum(x)/len(x) for x in zip(*square_error_list)]
