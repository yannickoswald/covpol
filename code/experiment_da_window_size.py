# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 14:28:32 2022

@author: earyo
"""


#%% Loading libraries

### RUN THIS SCRIPT TO REPRODUCE FIGURE 7

### import necessary libraries
#import os
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
from sklearn.linear_model import LinearRegression

#work laptop path
#os.chdir("C:/Users/earyo/Dropbox/Arbeit/postdoc_leeds/ABM_python_first_steps/implement_own_covid_policy_model")


#%% READ DATA
### read country/agent data
with open('./data/agent_data_v2.csv') as f:
    agent_data = pd.read_csv(f, encoding='unicode_escape')
Num_agents = len(agent_data)
agent_data["gdp_pc"] = pd.to_numeric(agent_data["gdp_pc"])

##### Read data for calibration
#### aggregate diffusion curve data

with open('./data/lockdown_diffusion_curve_updated_for_calibration.csv') as f:
    lockdown_data1 = pd.read_csv(f, encoding='unicode_escape', header=None)

#### data per country
with open('./data/lockdown_tracking.csv') as f:
    lockdown_data2 = pd.read_csv(f, encoding='unicode_escape')


#%% RUN PARTICLE FILTER EXPERIMENTS

### 40 corresponds to none
window_size_list = [40, 15, 10, 5, 2, 1]
da_instances_list = [int(30 / x) for x in window_size_list]

list_mse_results = []
list_micro_validity_results = []

number_of_particles = 1000

if __name__ == "__main__":

    for x in window_size_list:

        start = dt.now()

        pf_parameters = {
            "da_window": x,
            "da_instances": 30/x,
            "No_of_particles": number_of_particles
        }

        model_parameters = {
            "base_alert": 0.01,
            "social_base_threshold": 0.13,
            "clique_size": 18,
            "initial_conditions": 'real',
            "data_update": 'no'
        }

        current_PF = ParticleFilter(
            CountryModel, model_parameters, pf_parameters)
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

        """
        def create_fanchart_PF(arr):
            x = np.arange(arr.shape[0]) + 1
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
                ax.fill_between(x, low, high, color='tab:blue', alpha=alpha)
            
            ax.set_xlabel("Day of March")
            ax.set_ylabel("% of countries in lockdown")
            ax.legend(['Mean'] + [f'Pct{int(2*o)}' for o in offsets] + ['data'], frameon = False)
            ax.margins(x=0)
            return fig, ax
        
        create_fanchart_PF(results_pf/Num_agents*100)
        plt.savefig('fanchart_1_macro_validity_PF.png', bbox_inches='tight', dpi=300)
        plt.show()
        """

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
        microvalidity_mean = np.mean(micro_validity_metric_array_pf, axis=1)

        """plt.plot(np.linspace(1,31,31), mse_pf)
        plt.xlabel("Day of March")
        plt.ylabel("Mean squared error") ### perhaps plot squared error as fan-chart around?
        plt.savefig('MSE_over_time.png', bbox_inches='tight', dpi=300)
        plt.show()"""

        list_mse_results.append(sum(mse_pf))
        list_micro_validity_results.append(microvalidity_mean)

    df_mse = pd.DataFrame(list_mse_results)
    df_mv = pd.DataFrame(list_micro_validity_results)

    #%%

    #### plot data
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    #labels = ["h", "No PF", "15", "10", "5", "1"]

    labels = [str(x) for x in da_instances_list]

    labels.insert(0, "h")

    x = np.expand_dims(np.array(da_instances_list), 1)

    y = np.array(df_mse)

    reg = LinearRegression().fit(x, y)

    x_range = np.expand_dims(np.linspace(0, 30, 31), 1)
    y_predict = reg.coef_ * x_range + reg.intercept_

    rounded_a = float('%.3g' % reg.coef_)
    rounded_b = float('%.3g' % reg.intercept_)

    ax1.scatter(da_instances_list, df_mse, label="data")
    ax1.set_ylim((0, float(df_mse.max() * 1.1)))
    ax1.set_ylabel("sum of MSE over time")
    ax1.set_xlabel("Data assimilation frequency over 31 days")
    #ax1.set_xticklabels(labels)
    ax1.text(-0.5, float(df_mse.max() * 1.1) * (1 + 0.05), 'a', fontsize=12)
    ax1.text(22, float(df_mse.max() * 0.95),
             f'N = {number_of_particles}', fontsize=12)
    ax1.text(0.01, 0.03, f' y = {rounded_a}x + {rounded_b}', fontsize=12)
    ax1.plot(np.linspace(0, 30, 31), y_predict,
             linestyle='--', c="tab:red", label="lin. fit")
    ax1.legend(frameon=False, bbox_to_anchor=[0.3, 0.3])

    for i in range(len(window_size_list)):
        ax2.plot(np.linspace(1, 31, 31), df_mv.iloc[i, :], label=labels[i+1])
    ax2.legend(frameon=False)
    ax2.margins(0)
    ax2.set_ylim((0.4, 1))
    ax2.text(0, 1 + 0.03, 'b', fontsize=12)
    ax2.set_ylabel("mean % of countries correctly estimated")
    ax2.set_xlabel("Data assimilation frequency over 31 days")

    plt.savefig('fig7.png', bbox_inches='tight', dpi=300)

    ##### also plot "polynomial regression in panel (a) with NO PF set as a value of 30
    ### because a 30 day interval is basically meaningless for our period of time considered

    ### save data
    df_mse.to_csv("df_mse_window_size_exp.csv", sep=',')
    df_mv.to_csv("df_mv_window_size_exp.csv", sep=',')
