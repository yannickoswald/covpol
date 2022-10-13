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

#######################
#### IMPORT MODEL #####
#######################

from model_class2 import CountryModel

class ParticleFilter():
    
    '''
    A particle filter to create n instances of a the CountryModel
    for Covid lockdown diffusion and run the filtering process with
    sequential importance resampling.
    
    '''

    def __init__(self, ModelClass:CountryModel, model_params:dict, filter_params:dict):
        
       '''
       Initialise Particle Filter
           
       DESCRIPTION
       This method defines a few class properties, setting the parameters
       for the particle filter, the model and the storing the model/particle data
       
       PARAMETERS
        - ModelClass:              The model class is set to CountryModel
        - No_of_particles          The number of particles used to simulate the model
        - da_window:               The number of days between filtering steps
        - da_instances   :         The number of times that filtering is undertaken

       '''
       
       ### GLOBAL PF parameters
       self.da_window = filter_params['da_window']
       self.da_instances = int(filter_params['da_instances'])
       self.No_of_particles = filter_params['No_of_particles']
       
       ### MODEL parameters given as one more PF parameter
       self.model_params = model_params
       
       ### PF Data variables
       self.part_filtered_all = []
       self.weights = []

    
    ### create and store particles 
    def create_particles(self):
        
        '''DESCRIPTION
           This method creates the particles based on
           No_of_particles with model parameters provided.
        
           PARAMETERS
           - self only '''
    
        list_of_particles = []
        for i in range(self.No_of_particles):
                ### call the model iteration
                ### the 4th parameter, the initial conditions, is a string
                ### and can be set to 'real', 'no countries yet' or 'random'
                current_model = CountryModel(
                                              self.model_params['base_alert'],
                                              self.model_params['social_base_threshold'],
                                              self.model_params['clique_size'],
                                              self.model_params['initial_conditions'],
                                              self.model_params['data_update']
                                              )
                                
                current_model.model_id = i
                list_of_particles.append(copy.deepcopy(current_model)) 
                
        return list_of_particles
                
    
    @classmethod
    def error_particle_obs(cls, particle):
        
        '''DESCRIPTION
           Returns a metric for the error between particle and real-world
           If close to 0 that means high error. If close to 1 this means 
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
    
    @classmethod
    def particle_weights(cls, error_list):
        
        '''DESCRIPTION
            ### Method that omputes all particle weights from particle error/validity metric
            ### squared to "penalize" low ranking particles more.
        
           PARAMETERS
           - error list:        the errors of the particles  '''

        weights = [error**2 for error in error_list]
        ### normalization constraint such that sum(weights) = 1
        weights = weights/sum(weights)    
        return weights
    
    
    @classmethod
    def resample_particles(cls, list_of_particles_arg, weights_arg):
        
        '''DESCRIPTION
            Method that resamples the particles after each data assimilation
            window. The function works based on sequential importance resampling,
            where every particle weight determines its likelihood to be 
            resampled. The weights are cumulatively counted, so they constitute a
            cumulative distr. function (CDF) -- a weight distribution. And this distr. is
            compared against a uniformly random partition of the interval [0,1]
            constitung a uniformly random CDF. If the uniformly random CDF 'makes' 
            larger steps than the weights cumulation, because the weights are small, 
            it is likely that particles are filtered out.
        
           PARAMETERS
           - list_of_particles_arg:        the errors of the particles 
           - weights_arg:                  list of weights     '''
        
        
        weights = weights_arg
        list_of_particles = list_of_particles_arg
        
        number_of_particles = len(list_of_particles)
        re_sampled_particles = np.zeros(number_of_particles)
        random_partition_one_to_zero = ((np.arange(number_of_particles)
                                     + np.random.uniform()) / number_of_particles)    
        cumsum = np.cumsum(weights)

        i, j = 0, 0
        while i < number_of_particles:
                    if random_partition_one_to_zero[i] < cumsum[j]:
                        re_sampled_particles[i] = j
                        i += 1
                    else:
                        j += 1
                        
        list_of_particles_new = [copy.deepcopy(list_of_particles[int(x)]) 
                                 for x in re_sampled_particles]
        return list_of_particles_new
    
    
            
    #### steps a model one time step forward in time  
    #### particle arg. is one specific model instance
    #### return statement necessary 
    
    @classmethod
    def advance_particle(cls, particle): 
    
        '''DESCRIPTION
            #### This method teps a model one time step forward in time  
            #### return statement necessary 
        
           PARAMETERS
           - particle:        one particle/model instance '''

        particle.step()
        return particle
    

    def run_particle_filter(self):
        
        '''DESCRIPTION
           This method actually runs the particle filter. It mostly calls other
           methods defined above.
        
           PARAMETERS
           - only self'''

        
        list_of_particles = self.create_particles()
        list_of_lists_particles = []
        list_of_lists_weights = []

        for i in range(31):                
                
                [ParticleFilter.advance_particle(x) for x in list_of_particles]
            
                if (i > 0) and (i % self.da_window == 0):  
                    list_of_errors = [ParticleFilter.error_particle_obs(x) 
                                      for x in list_of_particles]
                    #print("This is list of errors", list_of_errors)
                    weights = ParticleFilter.particle_weights(list_of_errors)
                    list_of_particles = ParticleFilter.resample_particles(list_of_particles
                                                                          , weights)
                    list_of_lists_weights.append(weights)
                list_of_lists_particles.append(list_of_particles)
                #print("this is time step", i)
                
    
        self.part_filtered_all = list_of_lists_particles
        self.weights = list_of_lists_weights
        
       

#%% RUN PARTICLE FILTER EXPERIMENTS
 

pf_parameters = {
  "da_window": 5,
  "da_instances": 30/5,
  "No_of_particles": 100
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



current_PF.part_filtered_all



#%% PLOTTING PARTICLE FILTER RESULTS 

 ### TO DO verify whether pf works correctly 
No_of_particles = pf_parameters['No_of_particles']
da_window = pf_parameters['No_of_particles']
results = np.zeros((31, No_of_particles))

Dit={}
time_steps = 31
for i in range(31):
    for j in range(No_of_particles):
        ### key is a tuple where i equals time step, j particle number and
        ### the third number the model id, initially unique, but later can
        ### be associated with several particles because they are resample 
        ### versions of one another. 
        key = (i,j, current_PF.part_filtered_all[i][j].model_id)
        value = current_PF.part_filtered_all[i][j].datacollector.get_agent_vars_dataframe()
        Dit[key] = value
        df = Dit[key]
        df = (df.reset_index(level=0)).reset_index(level = 0)
        results[i,j] = df[(df.Step == i)]["Lockdown"].sum()
        
        
        

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

create_fanchart_PF(results/Num_agents*100)
plt.savefig('fanchart_1_macro_validity_PF.png', bbox_inches='tight', dpi=300)
plt.show()


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


create_fanchart_2_PF(micro_validity_PF*100)
plt.savefig('fanchart_2_micro_validity_PF.png', bbox_inches='tight', dpi=300)
plt.show()
