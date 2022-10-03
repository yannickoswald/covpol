# -*- coding: utf-8 -*-
"""
Covid lockdown adoption model and particle filter
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

#%% READ DATA, DEFINE A FEW CLASS INDEPENDENT FUNCTIONS AND GLOBAL VARIABLES
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

### this a function from here
### https://www.geeksforgeeks.org/program-distance-two-points-earth/
### for calculating the distance between points on earth

def geo_distance(lat1, lat2, lon1, lon2):
        # The math module contains a function named
        # radians which converts from degrees to radians.
        lon1 = radians(lon1)
        lon2 = radians(lon2)
        lat1 = radians(lat1)
        lat2 = radians(lat2)
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * asin(sqrt(a))
        # Radius of earth in kilometers. Use 3956 for miles
        r = 6371
        # calculate the result
        return c * r
    
    
### compute ranges of agent properties (later important for normalization
## of distance measure in the compute distance function by the agent)
max_income = max(agent_data["gdp_pc"]) ##
min_income = min(agent_data["gdp_pc"])
max_politicalregime = max(agent_data["democracy_index"])
min_politicalregime = min(agent_data["democracy_index"])
range_income = max_income - min_income
range_politicalregime = max_politicalregime - min_politicalregime

## max distance between two points on earth =
## earth circumference divided by two
max_distance_on_earth = 40075.017/2

#%%

class CountryAgent(mesa.Agent):
    
    '''initialization values for agent parameters which will be (re)initialized 
     from data through the model class'''
     
    NO_LOCKDOWN = 0
    LOCKDOWN = 1
    income_begin = 0
    politicalregime_begin = 0
    latitude_begin = 0
    longitude_begin = 0
    social_thre_begin = 0
    own_thre_begin = 0

    ### all agents need to have access to information about all other agents
    ### hence this list of agent instances
    ### https://stackoverflow.com/questions/328851/printing-all-instances-of-a-class


    def __init__(self, unique_id, model, 
                 init_income = income_begin,
                 init_politicalregime = politicalregime_begin,
                 init_latitude = latitude_begin,
                 init_longitude = longitude_begin,
                 init_social_thre = social_thre_begin,
                 init_own_thre = own_thre_begin,
                 init_state = NO_LOCKDOWN,
                 minimum_difference = 1,
                 adoption_mode = "none",
                 name = "name",
                 code = "code",
                 clique_size = 0,
                 ):
       '''
       Initialise Agent class
            
       PARAMETERS
         - unique id:             A unique agent id used for tracking individual agents
         - model:                 Model instance that contains the agent
         - init_income:           National income per country that is read from data
         - init_politicalregime   Democracy index per country that is read from data
         - init_latitude          Latitude of a country's capital
         - init_longitude         Longitude of a country's capital
         - init_social_thre       Threshold below which an agent adopts
                                  lockdown out of social motives/mimicking
         - init_own_thre          Threshold below which an agent adopts
                                  lockdown out of self-initiative
         - init_state             Set all countries to no lockdown at first 
                                  (12 out 164 will then be set into lockdown
                                   through data)
         - minimum_difference     Key variable of the model in that defines the
                                  distance between some agent and all other agents
         - adoption_mode          Records whether agent adopt through social mode
                                  or own initiative
         - name                   Set to none so far but will carry country name
         - code                   Set to generic code but will carry three letter
                                  ISO 3166 country code.
         - clique_size            Set to zero but can be variably reset with the model
                                  Important parameter that specifies how many
                                  other countries/agent an agent takes into account
                                   
       DESCRIPTION
       Agent class that represent one country in the world per agent

       '''
        
       super().__init__(unique_id, model)
       self.income = init_income
       self.politicalregime = init_politicalregime
       self.latitude = init_latitude
       self.longitude = init_longitude
       self.social_thre = init_social_thre
       self.own_thre = init_own_thre
       self.minimum_difference = 1
       self.adoption_mode = "none"
       self.name = "name"
       self.code = "code"
       self.clique_size = 0
       self.state = init_state
       if self.social_thre <= 0.001:
              self.social_thre  = 0.01
              
   
    
    def compute_distance(self):
        
       '''
        Agent's 'cognitive' function for computing the similarity between 
        themselves and other agents
             
        PARAMETERS
          None
                                    
        DESCRIPTION
         The agent acccesses its own income, politicalregime and position variable
         and computes the distance to all other countries that are already in 
         lockdown. Then the agent sorts these distances and takes the average
         of the k smallest distances as a guideline for the decision to 
         introduce a lockdown or not.
        ...

       '''
        
       ### execute the following steps only if not implemented a lockdown yet
       if self.state == 0:
           
           ### find all countries that have implemented lockdown already (through list comprehension)
           ### and store in array "total_differences_array"
           ### also sort by total difference value in ascending order
           
           y1 = self.income
           y2 = self.politicalregime
           y3_1 = self.latitude
           y3_2 = self.longitude

           total_differences_array = np.sort(np.array(
               [  1/3 * abs(y1 - x.income) / range_income 
                + 1/3 * abs(y2 - x.politicalregime) / range_politicalregime
                + 1/3 * (geo_distance(y3_1, x.latitude,
                                      y3_2, x.longitude) / max_distance_on_earth)
                 for x in self.model.schedule.agents if x.state == 1]
                  )
               )
           
           ## set the perceived difference of the agent to the other agents with lockdown
           ## to the observed average of differences across a certain clique size
           ## this means that the agent needs to observe in n = clique_size agents a behaviour 
           ### and computes how similar all of them (by taking the average) to onself are
           self.minimum_difference = np.mean(total_differences_array[0:self.clique_size])
           
           
       else:
             pass
    
    def update_state(self):
        
           #### DECISION THROUGH INITIATIVE
        
           if random.random() < self.own_thre: 
               self.state = 1
               self.adoption_mode = "initiative"   
               
           #### DECISION THROUGH SOCIAL MIMICKING
           
           ### wtl = (social) willingness to lockdown  
           ### add some modest stochasticity to the social willingness to locktown
           wtl = self.minimum_difference + np.random.normal(0, 0.005)   
           ### noise can make the wtl variable smaller than zero this why control
           ### flow here
           if wtl > 0: 
               if wtl < self.social_thre:
                   self.state = 1
                   self.adoption_mode = "social"
           else: 
               pass
               

    def step(self):
       self.compute_distance()
       if self.state == 0:       
           self.update_state()
           
       list_of_lockdown_countries = []
       list_of_lockdown_countries = [x for x in self.model.schedule.agents if x.state == 1]
       
       #### final global nudge to lockdown adoption in form a*e^b
       a = 0.00003
       b = 0.044*len(list_of_lockdown_countries)
       nudge_final = a*np.exp(b)    
       self.own_thre = self.own_thre + nudge_final
    
    def reset(cls):
        CountryAgent.instances = []
    


class CountryModel(mesa.Model):
    
    """A model with some number of agents."""
    def __init__(self, base_alert, social_base_threshold, 
                 clique_size, initial_conditions, data_update):
        self.num_agents = len(agent_data)
        self.base_alert = base_alert
        self.social_base_threshold = social_base_threshold
        self.initial_conditions = initial_conditions
        self.schedule = mesa.time.RandomActivation(self)
        self.running = True
        self.clique_size = clique_size
        self.time = -1 
        self.data_update = data_update
        self.model_id = 0 
        # Create agents based on external data
        ## or random initial conditions 
        ## or no countries at all in lockdown yet
        for i in range(self.num_agents):
            a = CountryAgent(i, self)
            self.schedule.add(a)
            
            if self.initial_conditions == "real":
                 a.state = agent_data["initial_state"][i]
            elif self.initial_conditions == "random":
                        if self.random.random() < 0.07:
                                               a.state = 1
            elif self.initial_conditions == "no countries yet":
                pass 
            else:
                sys.exit("No valid initial conditions supplied")
            
            a.name = agent_data["entity"][i]   
            a.income = agent_data["gdp_pc"][i]
            a.politicalregime = agent_data["democracy_index"][i]
            a.latitude = agent_data["capital_latitude"][i]
            a.longitude = agent_data["capital_longitude"][i]
            #### based on Sebhatu et al. (2020) population dense countries are
            #### more likely to introduce lockdown, democracies less so
            x = agent_data["log_population_density_normalized_on_average"][i]
            y = agent_data["democracy_index_normalized_on_average"][i]
            z = base_alert
            a.own_thre = (x**2) * (1/y) * z
            #### but the more democratic a country,
            #### the more sensitive they are to the influence of others 
            x1 = agent_data["democracy_index_normalized_on_average"][i]
            y1 = social_base_threshold
            a.social_thre = x1 * y1
            a.clique_size = self.clique_size
            a.code = agent_data["code"][i]
        
        self.datacollector = mesa.DataCollector(
            
            model_reporters={
                             ### lambda is an anonymous function
                             "N": lambda model: model.schedule.get_agent_count(),

                             },
            agent_reporters={ "code": "code",
                             "minimum_difference": "minimum_difference",
                             "Lockdown": "state",
                             "income": "income",
                             "politicalregime": "politicalregime",
                             "social_thre": "social_thre",
                             "own_thre": "own_thre",
                             "adoption_mode": "adoption_mode"
                             }
        )
    
    def step(self):
        
        self.time = self.time + 1
        
        
        if self.data_update == "yes" and i % 15 == 0:
             for agent in self.schedule.agents:
                 agent.state = lockdown_data2[(agent.code==lockdown_data2.Code) 
                                                                   & (self.time == lockdown_data2.model_step)]["lockdown"].iloc[0]
        else:
             pass

        self.datacollector.collect(self)
        self.schedule.step()
        
        
    #https://stackoverflow.com/questions/12179271/meaning-of-classmethod-and-staticmethod-for-beginner
    #@classmethod
    #def set_random_seed(cls, seed=None):
     #      '''Set a new numpy random seed
      #     :param seed: the optional seed value (if None then
       #    get one from os.urandom)
        #   '''
         #  new_seed = int.from_bytes(os.urandom(4), byteorder='little')\
          #     if seed is None else seed
          # np.random.seed(new_seed)

        
#%%
### run the model and save data

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


# parameters
## N = number of particles
## da_window = data_assimilation_window
### GLOBAL PF parameters
#percentage_filtered = 10
da_window = 5
da_instances = int(30/da_window)
No_of_particles = 100
#n_of_filterings_per_step = int(No_of_particles * percentage_filtered/100)


### create and store particles 

def create_particles(No_of_particles):

    list_of_particles = []
    for i in range(No_of_particles):
            ### call the model iteration
            ##4th parameter initial conditions can be real, no countries yet or random
            current_model = (CountryModel(0.01, 0.13, 18, 'real', 'no'))
            current_model.model_id = i
            #current_model.set_random_seed()
            list_of_particles.append(copy.deepcopy(current_model)) 
            
    return list_of_particles
            

list_of_particles = create_particles(No_of_particles)


def error_particle_obs(particle):
    '''DESCRIPTION
       returns a metric for the error between particle and real-world
       Counter-intuitively, if close to 0 that means high error. If close to 1 this means 
       low error. Because it measures the % of countries estimated in their correct state
    
       PARAMETERS
       particle because a specific particle needs to be passed  '''
    
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




particle = list_of_particles[i]


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
                    
    list_of_particles_new = [list_of_particles[int(x)] for x in re_sampled_particles]
    return list_of_particles_new, re_sampled_particles, weights










def advance_particle(particle):    
           for j in range(da_window):             
                            particle.step()


def run_particle_filter(da_instances_arg):
    da_instances = da_instances_arg
    for k in range(da_instances):
        list_of_errors = []        
        for i in range(len(list_of_particles)):
                advance_particle(list_of_particles[i])
                list_of_errors.append(error_particle_obs(list_of_particles[i]))
                
        
        weights = particle_weights(list_of_errors)
        list_of_particles_resampled = resample_particles(list_of_particles, weights)
        
    return list_of_particles_resampled
    
    
    



results_pf_test = np.zeros((da_window, No_of_particles))
results_pf_test_resampled = np.zeros((da_window, No_of_particles))
for i in range(No_of_particles):
   test = copy.deepcopy(list_of_particles[i].datacollector.get_agent_vars_dataframe())
   test2 = copy.deepcopy((test.reset_index(level=0)).reset_index(level = 0))
   
   test_re = copy.deepcopy(list_of_particles_filtered[i].datacollector.get_agent_vars_dataframe())
   test2_re = copy.deepcopy((test_re.reset_index(level=0)).reset_index(level = 0))
   print(list_of_particles[i].model_id)
   print(list_of_particles_filtered[i].model_id)
   for j in range(da_window):
       results_pf_test[j,i] = test2[(test2.Step == j)]["Lockdown"].sum()
       results_pf_test_resampled[j,i] = test2_re[(test2_re.Step == j)]["Lockdown"].sum()
       
       
plt.plot(results_pf_test)
plt.show()

plt.plot(results_pf_test_resampled)
plt.show()


