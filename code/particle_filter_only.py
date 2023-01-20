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
from particle_filter_class_parallelized import ParticleFilter
from run_base_model_opt import model_run
from multiprocessing import Pool


#work laptop path
os.chdir("C:/Users/earyo/Dropbox/Arbeit/postdoc_leeds/ABM_python_first_steps/implement_own_covid_policy_model/code")
    
if __name__ == "__main__":
    #%% READ DATA
    ### read country/agent data
    with open('C:/Users/earyo/Dropbox/Arbeit/postdoc_leeds/ABM_python_first_steps/implement_own_covid_policy_model/data/agent_data_v2.csv') as f:
        agent_data = pd.read_csv(f, encoding='unicode_escape')
    Num_agents = len(agent_data)
    agent_data["gdp_pc"] = pd.to_numeric(agent_data["gdp_pc"])

    ##### Read data for calibration
    #### aggregate diffusion curve data
    with open('C:/Users/earyo/Dropbox/Arbeit/postdoc_leeds/ABM_python_first_steps/implement_own_covid_policy_model/data/lockdown_diffusion_curve_updated_for_calibration.csv') as f:
        lockdown_data1 = pd.read_csv(f, encoding='unicode_escape', header=None)

    #### data per country
    with open('C:/Users/earyo/Dropbox/Arbeit/postdoc_leeds/ABM_python_first_steps/implement_own_covid_policy_model/data/lockdown_tracking.csv') as f:
        lockdown_data2 = pd.read_csv(f, encoding='unicode_escape')

    #%% RUN PARTICLE FILTER EXPERIMENTS
    start = dt.now()

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

    end = dt.now()
    time_elapsed = end - start
    print("particle filter takes", time_elapsed, " seconds")

    #%% PLOTTING PARTICLE FILTER RESULTS

    ### TO DO verify whether pf works correctly
    No_of_particles = pf_parameters['No_of_particles']
    da_window = pf_parameters['No_of_particles']
    results_pf = np.zeros((31, No_of_particles))
    micro_validity_metric_array_pf = np.zeros((31, No_of_particles))

    start = dt.now()
    #Dit={}
    time_steps = 31
    for i in range(31):
        micro_state_data = pd.Series.reset_index(
            lockdown_data2[(lockdown_data2.model_step == i)]["lockdown"], drop=True)
        for j in range(No_of_particles):
            ### key is a tuple where i equals time step, j particle number and
            ### the third number the model id, initially unique, but later can
            ### be associated with several particles because they are resampled
            ### versions of one another. Therefore ultimately j is the *unique*
            ### identifier as well as the tuple as a whole given j.
            #key = (i,j, current_PF.part_filtered_all[i][j].model_id)
            #value = current_PF.part_filtered_all[i][j].datacollector.get_agent_vars_dataframe()
            #Dit[key] = value
            df = current_PF.part_filtered_all[i][j].datacollector.get_agent_vars_dataframe(
            )
            df = (df.reset_index(level=0)).reset_index(level=0)
            results_pf[i, j] = df[(df.Step == i)]["Lockdown"].sum()
            micro_state = pd.Series.reset_index(
                df[(df.Step == i)]["Lockdown"], drop=True)
            micro_validity_metric_array_pf[i, j] = np.mean(
                micro_state == micro_state_data)

        print("the time step considered in data prep is ", i)

    end = dt.now()
    time_elapsed = end - start
    print("data writing takes", time_elapsed, " seconds")

    def create_fanchart_PF(arr):
        x = np.arange(arr.shape[0]) + 1
        # for the median use `np.median` and change the legend below
        mean = np.mean(arr, axis=1)
        offsets = (25, 67/2, 47.5)
        fig, ax = plt.subplots()
        ax.plot(x, mean, color='black', lw=3)
        for offset in offsets:
            low = np.percentile(arr, 50-offset, axis=1)
            high = np.percentile(arr, 50+offset, axis=1)
            # since `offset` will never be bigger than 50, do 55-offset so that
            # even for the whole range of the graph the fanchart is visible
            alpha = (55 - offset) / 100
            ax.fill_between(x, low, high, color='tab:blue', alpha=alpha)

        ax.set_xlabel("Day of March")
        ax.set_ylabel("% of countries in lockdown")
        ax.legend(
            ['Mean'] + [f'Pct{int(2*o)}' for o in offsets] + ['data'], frameon=False)
        ax.margins(x=0)
        return fig, ax

    create_fanchart_PF(results_pf/Num_agents*100)
    plt.savefig('fanchart_1_macro_validity_PF.png',
                bbox_inches='tight', dpi=300)
    plt.show()

    ##### PLOT mean squared error per time step. pf results vs no pf results
    results_pf_percent = results_pf/164
    square_diffs_pf = np.zeros((31, No_of_particles))
    for i in range(No_of_particles):
        square_diffs_pf[:, i] = (
            results_pf_percent[:, i] - lockdown_data1.iloc[:, 0].to_numpy())**2

    mse_pf = np.mean(square_diffs_pf, axis=1)

    results_pf_percent = results_pf/164

    square_diffs_pf = np.zeros((31, No_of_particles))
    square_diffs = np.zeros((31, No_of_particles))
    for i in range(No_of_particles):
        square_diffs_pf[:, i] = (
            results_pf_percent[:, i] - lockdown_data1.iloc[:, 0].to_numpy())**2

    mse_pf = np.mean(square_diffs_pf, axis=1)

    plt.plot(np.linspace(1, 31, 31), mse_pf)
    plt.xlabel("Day of March")
    # perhaps plot squared error as fan-chart around?
    plt.ylabel("Mean squared error")
    plt.savefig('MSE_over_time.png', bbox_inches='tight', dpi=300)
    plt.show()

    mse_list = [sum(mse_pf)]
    df_mse = pd.DataFrame(mse_list)

    df_mse.to_csv("df_mse.csv", sep=',')
