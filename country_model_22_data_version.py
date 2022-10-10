# -*- coding: utf-8 -*-
"""
Covid lockdown adoption model script and particle filter
modified: 08/09/2022
modified and created by: Yannick Oswald
"""

#%%

### import necessary libraries
import os
import mesa
import mesa.time
import mesa.space
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math as math
import mesa.batchrunner
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

#work laptop path
os.chdir("C:/Users/earyo/Dropbox/Arbeit/postdoc_leeds/ABM_python first steps/implement own covid policy model")


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
### import model class (which itself imports agent class)

from model_class2 import CountryModel
        
#%%
### run the model, without particle filter, and save data

start = dt.now()

no_of_iterations = 10
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
        
        print("model iteration is " + str(j))
        
        #CountryAgent.reset(CountryAgent)


running_secs = (dt.now() - start).seconds
print("running time was " + str(running_secs) + " sec")


print(model.schedule.agents)

#%%

#### PLOTTING

### Initial conditions 

#### plot #0.0 distributions of variables (income, democracy index, latitude and longitude)
### plot #0.1 map of lockdowns 

df_initial_conditions_countries = pd.DataFrame(columns = df_results.columns)

for i in range(no_of_iterations):
          df_initial_conditions_countries = pd.concat([df_initial_conditions_countries, 
                                                       df_results[(df_results.iteration == i) & (df_results.Step == 0) & (df_results.Lockdown == 1)]],
                                                       ignore_index=True)

### plot #1 number of lockdowns over time steps

###plotting takes too long if too many runs 
if no_of_iterations <= 200:

    fig1, ax1 = plt.subplots(figsize = (6,5))
    
    for i in range(no_of_iterations):
        ax1.plot( df_results[(df_results.iteration == i) & (df_results.AgentID == 0)]["Step"] + 1,
                  np.sum(np.split(np.array(df_results[(df_results.iteration == i)]["Lockdown"]),31),axis=1) / Num_agents * 100, alpha = 0.5)
    ax1.plot(df_results[(df_results.iteration == i) & (df_results.AgentID == 0)]["Step"] + 1, 
             lockdown_data1[0]*100,
             linewidth=3 ,label = "data")
    ax1.set_xlabel("Day of March 2020")
    ax1.set_ylabel("% of countries in lockdown")
    ax1.legend(frameon=False)
    ax1.text(23, 20, "Clique size: " + str(model.clique_size))
    ax1.text(23, 15, "Base alert: " + str(model.base_alert))
    ax1.text(23, 10, "Social alert: " + str(model.social_base_threshold))
    ax1.text(21, 5, "Initial conditions " + str(model.initial_conditions))
    plt.show()



### plot #1.1 distribution of runs at every time step
### is the distribution of the model estimate normal or not?
### df_results_filtered
df_results_filtered = df_results[(df_results.AgentID == 9)]


array_run_results = np.zeros((no_of_iterations,31))
for i in range(no_of_iterations):
    array_run_results[i,:] = np.sum(np.split(np.array(df_results[(df_results.iteration == i)]["Lockdown"]),31),axis=1)


###plotting takes too long if too many runs and also does not make sense if too few
if no_of_iterations >= 10 and no_of_iterations <= 200:
    
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    spec = fig.add_gridspec(ncols=6, nrows=6)
    
    for i in range(31):
        globals()['ax11%s' % i]  = fig.add_subplot(spec[math.floor(i/6), i%6])
        globals()['ax11%s' % i].hist(array_run_results[:,i], bins = int(no_of_iterations/10))
        globals()['ax11%s' % i].set_xlim([0, 164])
        #globals()['ax11%s' % i].set_ylim([0, no_of_iterations])
    fig.suptitle('Distribution of run results at every time step')
    #https://stackoverflow.com/questions/6986986/bin-size-in-matplotlib-histogram
    plt.savefig('Distribution of runs over each time step.png', bbox_inches='tight', dpi=300)

### plot #2 average minimum_difference (should decay over time)
### because more countries adopt a lockdown so for each country more and more similar countries 
### serve as a benchmark

if no_of_iterations <= 200:

    fig2, ax2 = plt.subplots(figsize = (6,5))
    
    for i in range(no_of_iterations):
        df_iter = df_results[(df_results.iteration == i)]
        average_min_diff_array = np.zeros((31,1))
        average_min_and_max_diff = np.zeros((31,2))
        for j in range(31):
               average_min_diff_array[j,0] = np.mean( df_iter[(df_iter.Step == j)]["minimum_difference"])
               average_min_and_max_diff[j,0] = min(df_iter[(df_iter.Step == j)]["minimum_difference"])
               average_min_and_max_diff[j,1] = max(df_iter[(df_iter.Step == j)]["minimum_difference"])
                             
        ax2.plot((df_results[(df_results.iteration == i) & (df_results.AgentID == 0)]["Step"] + 1)[1:30], average_min_diff_array[1:30])
        #plt.plot((df_results[(df_results.iteration == i) & (df_results.AgentID == 0)]["Step"] + 1)[1:30], average_min_and_max_diff[1:30,0])
        #plt.plot((df_results[(df_results.iteration == i) & (df_results.AgentID == 0)]["Step"] + 1)[1:30], average_min_and_max_diff[1:30,1])
           
    ax2.set_xlabel("Day of March 2020")
    ax2.set_ylabel("Average min distance")
    plt.show()

### plot #3 plots the micro-validity (or non-validity of the model) by
### measuring the difference between the day a country adopts in the model and adopted in the real-world
### (thus the difference is measured in number of days)
### if number is positive it is too late
### if negative, it is too early predicted



micro_validity_BIG = np.zeros((no_of_iterations,164))
if no_of_iterations <= 100:
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    for i in range(Num_agents):  
            model_lockdown_data = df_results[(df_results.code == agent_data["code"][i]) & (df_results.Lockdown == 1)]
            real_world_lockdown_date = lockdown_data2[(lockdown_data2.Code ==  agent_data["code"][i]) & (lockdown_data2.lockdown == 1)]
            for p in range(no_of_iterations):
                model_lockdown_data2 = model_lockdown_data[(model_lockdown_data.iteration == p)]["Step"]
                if len(model_lockdown_data2) > 0 and len(real_world_lockdown_date) > 0:
                    difference = model_lockdown_data2.iloc[0] - (int(real_world_lockdown_date.iloc[0]["Day"][0:2])-1)
                    micro_validity_BIG[p,i] = difference 
                    ax3.scatter(i,difference, color = "tab:blue", alpha = 0.2)
            print("iteration of micro-level-plot is " + str(i))
    ax3.set_ylabel("diff. in days")   
    ax3.set_xlabel("country index")  
    ax3.plot([0,164],[0,0], color = "black", linewidth = 3)
    ax3.margins(0)
    plt.savefig('Micro_validity_1.png', bbox_inches='tight', dpi=300)
    plt.show()



### plot #3 NEW BETTER VERSION


df_differences_per_country = pd.DataFrame(data=micro_validity_BIG.T,
                 index=agent_data.code , 
                 columns=np.linspace(0,no_of_iterations-1,no_of_iterations).astype(int),
                 dtype=None, copy=None)

micro_validity_BIG_sq_diff_sum = np.sum((micro_validity_BIG - 0)**2, axis = 0)
df_differences_per_country = pd.concat([df_differences_per_country,
                                        pd.DataFrame(micro_validity_BIG_sq_diff_sum, 
                                                     index = agent_data.code, columns = ["error"])],
                                       axis = 1 )

df_differences_per_country =  pd.concat([df_differences_per_country,
                                         pd.DataFrame(np.array(agent_data["gdp_pc"]), index = agent_data.code),
                                         pd.DataFrame(np.array(agent_data["democracy_index"]), index = agent_data.code)], axis = 1)

df_differences_per_country_sorted = df_differences_per_country.sort_values(by=['error'])

## https://stackoverflow.com/questions/25408393/getting-individual-colors-from-a-color-map-in-matplotlib 
## colormap legend

cmap = plt.cm.viridis
norm = mpl.colors.Normalize(vmin=0, vmax=10)

fig30, ax30 = plt.subplots(figsize=(12, 6))
for i in range(164):
    ax30.scatter(np.repeat(i, len(df_differences_per_country_sorted.iloc[i,0:no_of_iterations])),
                 df_differences_per_country_sorted.iloc[i,0:no_of_iterations], 
                 color = cmap(df_differences_per_country_sorted.iloc[i,12]/10), alpha = 0.2)    
    ax30.set_ylabel("diff. in days", size = 16)   
    ax30.set_xlabel("country index", size = 16) 
    ax30.plot([0,164],[0,0], color = "black", linewidth = 3)
    ax30.margins(x=0)
#ax30.set_xticks(np.linspace(1,164,164), df_differences_per_country_sorted.index, rotation=90, size = 8)
ax30.xaxis.set_tick_params(labelsize=16)
ax30.yaxis.set_tick_params(labelsize=16)

cbar_ax = fig30.add_axes([0.35, 0.21, 0.15, 0.03])
#fig30.colorbar(ax30, cax=cbar_ax)
fig30.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cbar_ax, orientation='horizontal', label='Democracy index')

plt.savefig('Micro_validity_2.png', bbox_inches='tight', dpi=300)


### plot #4 fan chart of model runs 
##https://stackoverflow.com/questions/28807169/making-a-python-fan-chart-fan-plot
# THIS --> https://stackoverflow.com/questions/66146705/creating-a-fanchart-from-a-series-of-monte-carlo-projections-in-python


def create_fanchart(arr):
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
    
    ax.plot(df_results[(df_results.iteration == 0) & (df_results.AgentID == 0)]["Step"] +1, 
             lockdown_data1[0]*100,
             linewidth=3 ,label = "data", linestyle= "--", color = "tab:red")
    
    ax.set_xlabel("Day of March")
    ax.set_ylabel("% of countries in lockdown")
    ax.legend(['Mean'] + [f'Pct{int(2*o)}' for o in offsets] + ['data'], frameon = False)
    ax.margins(x=0)
    return fig, ax

create_fanchart(array_run_results.T/Num_agents*100)
plt.savefig('fanchart_1_macro_validity.png', bbox_inches='tight', dpi=300)
plt.show()
#4.1 report least squares of mean to data and variance per time step 
## (both metrics need to minimized)
mean_model_runs = np.mean(array_run_results, axis = 0)
variance_model_runs = np.var(array_run_results, axis = 0)

### plot #5 micro validity over time as a function of how many countries are in their
### correct lockdown state (lockdown yes or no)

### test whether dataframes are in the exact same order
## test = df_results[(df_results.iteration == i) & (df_results.Step == j)]["code"] 
### == pd.Series.reset_index(lockdown_data2[(lockdown_data2.model_step == j)]["Code"], drop=True)

micro_validity_metric_array = np.zeros((31,no_of_iterations))
   
for i in range(no_of_iterations):
       micro_validity_metric_array[:,i] = np.mean(
                                                    np.array_split(
                                                                    np.array(
                                                                             pd.Series.reset_index(
                                                                                 df_results[(df_results.iteration == i) ]["Lockdown"], drop = True) 
                                                                             
                                                                             == 
                                                                                   
                                                                             pd.Series.reset_index(
                                                                                 lockdown_data2.sort_values(
                                                                                                   ['model_step', 'Entity']
                                                                                                   )
                                                                                           ["lockdown"],drop = True
                                                                                      )
                                                                           ),
                                                                31)
                                                 ,axis = 1
                                                )





squared_error_macro_level = ((mean_model_runs/164 - lockdown_data1[0].to_numpy())**2)*100
plt.scatter(np.log(np.var(micro_validity_metric_array, axis =1 )), np.log(squared_error_macro_level))
plt.xlabel("log(micro_validity_metric_variance)")
plt.ylabel("log(squared_error_macro_level)")
plt.title("Predicting the accuracy of the model macro level from micro pattern")
mean_squared_error_macro_level = np.mean(squared_error_macro_level)



mean_at_t_15 = mean_model_runs[15]/164
datapoints_at_t_15 = array_run_results[:,15]/164



def create_fanchart_2(arr):
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


create_fanchart_2(micro_validity_metric_array*100)
plt.savefig('fanchart_2_micro_validity.png', bbox_inches='tight', dpi=300)
plt.show()

#%%

non_adopters = df_results[(df_results.iteration == 0) & (df_results.Step == 30) & (df_results.Lockdown == 0)]

#%%

####################################
####################################
######   Particle Filter      ######
####################################
####################################


### role-model?
### https://github.com/Urban-Analytics/dust/blob/main/Projects/ABM_DA/stationsim/particle_filter_gcs.py



### GLOBAL PF parameters

#### da_window determines how many time steps prediction is made without DA
da_window = 5
da_instances = int(30/da_window)
No_of_particles = 100
#n_of_filterings_per_step = int(No_of_particles * percentage_filtered/100)


### create and store particles 

def create_particles(No_of_particles):

    list_of_particles = []
    for i in range(No_of_particles):
            ### call the model iteration
            ### the 4th parameter, the initial conditions, is a string
            ### and can be set to 'real', 'no countries yet' or 'random'
            current_model = (CountryModel(0.01, 0.13, 18, 'real', 'no'))
            current_model.model_id = i
            list_of_particles.append(copy.deepcopy(current_model)) 
            
    return list_of_particles
            


def error_particle_obs(particle):
    
    '''DESCRIPTION
       returns a metric for the error between particle and real-world
       Counter-intuitively, if close to 0 that means high error. If close to 1 this means 
       low error. Because it measures the % of countries estimated in their correct state
    
       PARAMETERS
       - particle:         because a specific particle needs to be passed  '''
    
    ## find current time step
    t = particle.time
    ### go through all datasteps necessary to find error
    data1 = copy.deepcopy(particle.datacollector.get_agent_vars_dataframe())
    data2 = (data1.reset_index(level=0)).reset_index(level = 0)
    data3 = data2[(data2.Step == t)]
    data4 = pd.Series.reset_index(data3["Lockdown"], drop = True)
    data5 = lockdown_data2[(lockdown_data2.model_step == t)]["lockdown"]
    data6 = pd.Series.reset_index(data5, drop = True)
    particle_validity = np.mean(data4 == data6)
    return particle_validity



##unit test here? 
### what happens if i give a certain num. like 1 to the weights
### test the function itself

### https://github.com/Urban-Analytics/RAMP-UA/blob/master/tests/test_microsim_model.py

##  pytest


def particle_weights(error_list):
    ### DESCRIPTION
    ### computes all particle weights from particle error/validity metric
    ### squared to "penalize" low ranking particles more
    weights = [error**2 for error in error_list]
    ### normalization constraint such that sum(weights) = 1
    weights = weights/sum(weights)    
    return weights


##unit test here? 

def resample_particles(list_of_particles_arg, weights_arg):
    
    weights = weights_arg
    list_of_particles = list_of_particles_arg
    
    number_of_particles = len(list_of_particles)
    re_sampled_particles = np.zeros(number_of_particles)
    random_partition_one_to_zero = ((np.arange(number_of_particles)
                                 + np.random.uniform()) / number_of_particles)

    cumsum = np.cumsum(weights)
    i, j = 0, 0
    while i < number_of_particles :
                if random_partition_one_to_zero[i] < cumsum[j]:
                    re_sampled_particles[i] = j
                    i += 1
                else:
                    j += 1
                    
    list_of_particles_new = [copy.deepcopy(list_of_particles[int(x)]) for x in re_sampled_particles]
    return list_of_particles_new



def advance_particle(particle):                
                            particle.step()


def run_particle_filter(da_instances_arg, da_window_arg):
    
    list_of_particles = create_particles(No_of_particles)
    da_instances = da_instances_arg
    da_window = da_window_arg
    
    for k in range(da_instances + 1):
        
            if k < da_instances:
                
                list_of_errors = []        
                for i in range(len(list_of_particles)):
                    
                                  for j in range(da_window):
                                        advance_particle(list_of_particles[i])
                                  assert list_of_particles[i].time <= 30
                                  list_of_errors.append(error_particle_obs(list_of_particles[i]))
                                                
                                    
                weights = particle_weights(list_of_errors)
                list_of_particles = resample_particles(list_of_particles, weights)
                
            else: 
                
                list_of_errors = []        
                for i in range(len(list_of_particles)):
                    
                              for j in range(1):
                                    advance_particle(list_of_particles[i])
                              assert list_of_particles[i].time <= 30
                              list_of_errors.append(error_particle_obs(list_of_particles[i]))
       
                weights = particle_weights(list_of_errors)
                list_of_particles = resample_particles(list_of_particles, weights)
                
                

    list_of_particles_filtered = list_of_particles
    return list_of_particles_filtered, weights
    
    
    


#%%


class ParticleFilter():
    
    '''
    A particle filter to model the dynamics of the
    state of the model as it develops in time.
    
    '''

    def __init__(self, ModelClass:CountryModel, model_params:dict, filter_params:dict):
        
       '''
       Initialise Particle Filter
           
       PARAMETERS
        - number_of_particles:     The number of particles used to simulate the model
        - number_of_runs:          The number of times to run this particle filter (e.g. experiment)
        - resample_window:         The number of iterations between resampling particles
        - multi_step:              Whether to do all model iterations in between DA windows in one go

       DESCRIPTION
       Firstly, set all attributes using filter parameters. Set time and
       initialise base model using model parameters. Initialise particle
       models using a deepcopy of base model. Determine particle filter 
       dimensions, initialise all remaining arrays, and set initial
       particle states to the base model state using multiprocessing. 
       '''
       
    ### GLOBAL PF parameters
    
    #### da_window determines how many time steps prediction is made without DA
    da_window = filter_params['da_window']
    da_instances = filter_params['da_instances']
    No_of_particles = filter_params['No_of_particles']
    
    
    ### create and store particles 
    
    def create_particles(No_of_particles):
    
        list_of_particles = []
        for i in range(No_of_particles):
                ### call the model iteration
                ### the 4th parameter, the initial conditions, is a string
                ### and can be set to 'real', 'no countries yet' or 'random'
                current_model = (CountryModel(
                                              model_params['base_alert'],
                                              model_params['social_base_threshold'],
                                              model_params['clique_size'],
                                              model_params['initial_conditions'],
                                              model_params['data_update']
                                              )
                                 )
                current_model.model_id = i
                list_of_particles.append(copy.deepcopy(current_model)) 
                
        return list_of_particles
                
    
    
    def error_particle_obs(particle):
        
        '''DESCRIPTION
           returns a metric for the error between particle and real-world
           Counter-intuitively, if close to 0 that means high error. If close to 1 this means 
           low error. Because it measures the % of countries estimated in their correct state
        
           PARAMETERS
           - particle:         because a specific particle needs to be passed  '''
        
        ## find current time step
        t = particle.time
        ### go through all datasteps necessary to find error
        data1 = copy.deepcopy(particle.datacollector.get_agent_vars_dataframe())
        data2 = (data1.reset_index(level=0)).reset_index(level = 0)
        data3 = data2[(data2.Step == t)]
        data4 = pd.Series.reset_index(data3["Lockdown"], drop = True)
        data5 = lockdown_data2[(lockdown_data2.model_step == t)]["lockdown"]
        data6 = pd.Series.reset_index(data5, drop = True)
        particle_validity = np.mean(data4 == data6)
        return particle_validity
    
    
    def particle_weights(error_list):
        ### DESCRIPTION
        ### computes all particle weights from particle error/validity metric
        ### squared to "penalize" low ranking particles more
        weights = [error**2 for error in error_list]
        ### normalization constraint such that sum(weights) = 1
        weights = weights/sum(weights)    
        return weights
    
    
    ##unit test here? 
    
    def resample_particles(list_of_particles_arg, weights_arg):
        
        weights = weights_arg
        list_of_particles = list_of_particles_arg
        
        number_of_particles = len(list_of_particles)
        re_sampled_particles = np.zeros(number_of_particles)
        random_partition_one_to_zero = ((np.arange(number_of_particles)
                                     + np.random.uniform()) / number_of_particles)
    
        cumsum = np.cumsum(weights)
        i, j = 0, 0
        while i < number_of_particles :
                    if random_partition_one_to_zero[i] < cumsum[j]:
                        re_sampled_particles[i] = j
                        i += 1
                    else:
                        j += 1
                        
        list_of_particles_new = [copy.deepcopy(list_of_particles[int(x)]) for x in re_sampled_particles]
        return list_of_particles_new
    
    
    
    def advance_particle(particle):                
                                particle.step()
    
    
    def run_particle_filter(da_instances_arg, da_window_arg):
        
        list_of_particles = create_particles(No_of_particles)
        da_instances = da_instances_arg
        da_window = da_window_arg
        
        for k in range(da_instances + 1):
            
                if k < da_instances:
                    
                    list_of_errors = []        
                    for i in range(len(list_of_particles)):
                        
                                      for j in range(da_window):
                                            advance_particle(list_of_particles[i])
                                      assert list_of_particles[i].time <= 30
                                      list_of_errors.append(error_particle_obs(list_of_particles[i]))
                                                    
                                        
                    weights = particle_weights(list_of_errors)
                    list_of_particles = resample_particles(list_of_particles, weights)
                    
                else: 
                    
                    list_of_errors = []        
                    for i in range(len(list_of_particles)):
                        
                                  for j in range(1):
                                        advance_particle(list_of_particles[i])
                                  assert list_of_particles[i].time <= 30
                                  list_of_errors.append(error_particle_obs(list_of_particles[i]))
           
                    weights = particle_weights(list_of_errors)
                    list_of_particles = resample_particles(list_of_particles, weights)
                    
                    
    
        list_of_particles_filtered = list_of_particles
        return list_of_particles_filtered, weights



#%%
#RUN PARTICLE FILTER
list_of_particles_filtered, weights = run_particle_filter(da_instances, da_window)    

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

#%% PLOTTING PARTICLE FILTER RESULTS
results_pf_test = np.zeros((da_window, No_of_particles))
results_pf_test_resampled = np.zeros((31, No_of_particles))
micro_validity_PF = np.zeros((31,No_of_particles))
for i in range(No_of_particles):

   test_re = copy.deepcopy(list_of_particles_filtered[i].datacollector.get_agent_vars_dataframe())
   test2_re = copy.deepcopy((test_re.reset_index(level=0)).reset_index(level = 0))

   print(list_of_particles_filtered[i].model_id)
   for j in range(31):
       results_pf_test_resampled[j,i] = test2_re[(test2_re.Step == j)]["Lockdown"].sum()
       
       model_micro = test2_re[(test2_re.Step == j)]["Lockdown"].reset_index(level = 0, drop = True)
       empirical_micro = lockdown_data2[(lockdown_data2.model_step == j)]["lockdown"].reset_index(level = 0, drop = True)
       
       micro_validity_PF[j,i] = (model_micro == empirical_micro).sum()/164
      


plt.plot(results_pf_test_resampled)
plt.show()



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
    
    ax.plot(df_results[(df_results.iteration == 0) & (df_results.AgentID == 0)]["Step"] +1, 
             lockdown_data1[0]*100,
             linewidth=3 ,label = "data", linestyle= "--", color = "tab:red")
    
    ax.set_xlabel("Day of March")
    ax.set_ylabel("% of countries in lockdown")
    ax.legend(['Mean'] + [f'Pct{int(2*o)}' for o in offsets] + ['data'], frameon = False)
    ax.margins(x=0)
    return fig, ax

create_fanchart_PF(results_pf_test_resampled/Num_agents*100)
plt.savefig('fanchart_1_macro_validity_PF.png', bbox_inches='tight', dpi=300)
plt.show()




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


create_fanchart_2_PF(micro_validity_PF*100)
plt.savefig('fanchart_2_micro_validity_PF.png', bbox_inches='tight', dpi=300)
plt.show()

