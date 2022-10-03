# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 09:49:13 2022

@author: earyo
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 14:27:46 2022
@author: earyo

"""

#import geopandas as gpd
#import fiona
#home laptop path
#os.chdir("C:/Users/y-osw/Dropbox/Arbeit/postdoc_leeds/ABM_python first steps/implement own covid policy model")

#work laptop path
#os.chdir("C:/Users/earyo/Dropbox/Arbeit/postdoc_leeds/ABM_python first steps/implement own covid policy model")
##https://geoffboeing.com/2014/09/using-geopandas-windows/

## https://stackoverflow.com/questions/50876702/cant-install-fiona-on-windows
#countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
#countries.head()



#%%
### import necessary libraries
import os
import mesa
import mesa.time
import mesa.space
import matplotlib.pyplot as plt
import numpy as np
import math as math
import mesa.batchrunner
import pandas as pd
import copy as copy
from math import radians, cos, sin, asin, sqrt
import random
from datetime import datetime as dt
import sys
#work laptop path
os.chdir("C:/Users/earyo/Dropbox/Arbeit/postdoc_leeds/ABM_python first steps/implement own covid policy model")

#%% READ DATA, DEFINE A FEW CLASS INDEPENDENT FUNCTIONS AND GLOBAL VARIABLES
### read country/agent data
agent_data = pd.read_csv('agent_data_v2.csv', encoding = 'unicode_escape')
Num_agents = len(agent_data)
agent_data["gdp_pc"] = pd.to_numeric(agent_data["gdp_pc"])

lockdown_data_for_calibration_agg = pd.read_csv('lockdown_diffusion_curve_updated_for_calibration.csv', 
                                                encoding = 'unicode_escape',
                                                header = None)

lockdown_data_for_calibration_micro = pd.read_csv('lockdown_tracking.csv', 
                                                encoding = 'unicode_escape')

day_to_steps = {"day": lockdown_data_for_calibration_micro["Day"][0:31],
                "step":np.linspace(0,30,31).tolist()}

df = pd.DataFrame(day_to_steps)


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
    
    """initialization values for agent properties which will be (re)set
     through the model class """
     
    NO_LOCKDOWN = 0
    LOCKDOWN = 1
    income_begin = 0
    politicalregime_begin = 0
    latitude_begin = 0
    longitude_begin = 0
    social_adoption_threshold_begin = 0
    alert_adoption_threshold_begin = 0
    
    
    ### all agents need to look have access to information about all other agents
    ### hence this list of agent instances
    ### https://stackoverflow.com/questions/328851/printing-all-instances-of-a-class
    instances = []
    
    "initialize agent with all properties"
    def __init__(self, unique_id, model, 
                 init_income = income_begin,
                 init_politicalregime = politicalregime_begin,
                 init_latitude = latitude_begin,
                 init_longitude = longitude_begin,
                 init_social_adoption_threshold = social_adoption_threshold_begin,
                 init_alert_adoption_threshold = alert_adoption_threshold_begin,
                 init_state=NO_LOCKDOWN):
        
       super().__init__(unique_id, model)
       self.income = init_income
       self.politicalregime = init_politicalregime
       self.latitude = init_latitude
       self.longitude = init_longitude
       self.social_adoption_threshold = init_social_adoption_threshold
       self.alert_adoption_threshold = init_alert_adoption_threshold
       self.minimum_difference = 1
       self.adoption_mode = "none"
       self.name = "name"
       self.code = "code"
       self.clique_size = 0
       self.state = init_state
       self.__class__.instances.append(self)
       if self.social_adoption_threshold <= 0.001:
              self.social_adoption_threshold  = 0.01
              
    
    # do something with the agent object
    def compute_distance(self):
        
       ### execute the following steps only if not implemented a lockdown yet
       if self.state == 0:
           
           ### find all countries that have implemented lockdown already (through list comprehension)
           ### and store in array "total_differences_array"
           ### also sort by total difference value in ascending order

           total_differences_array = np.sort(np.array(
               [  1/3 * abs(self.income - x.income) / range_income 
                + 1/3 * abs(self.politicalregime - x.politicalregime) / range_politicalregime
                + 1/3 * (geo_distance(self.latitude, x.latitude,
                                      self.longitude, x.longitude) / max_distance_on_earth)
                 for x in CountryAgent.instances if x.state == 1]))

           ## set the perceived difference of the agent to the other agents with lockdown
           ## to the observed average of differences across a certain clique size
           ## this means that the agent needs to observe in n = clique_size agents a behaviour 
           ### and computes how similar all of them (by taking the average) to onself are
           self.minimum_difference = np.mean(total_differences_array[0:self.clique_size])
           
           
       else:
             pass
    
    def update_state(self):
        
           if random.random() < self.alert_adoption_threshold: 
               self.state = 1
               self.adoption_mode = "initiative"
               
           wtl = self.minimum_difference + np.random.normal(0, 0.005)  ### wta = willingness to lockdown    
               
           if wtl > 0: 
               if wtl < self.social_adoption_threshold:
                   self.state = 1
                   self.adoption_mode = "social"
           else: 
               pass
               

    def step(self):
       self.compute_distance()
       
       if self.state == 0:       
           self.update_state()
           
       list_of_lockdown_countries = []
       list_of_lockdown_countries = [x for x in CountryAgent.instances if x.state == 1]
           
       self.alert_adoption_threshold = self.alert_adoption_threshold + (self.alert_adoption_threshold + 0.00003*np.exp(0.044*len(list_of_lockdown_countries)) - self.alert_adoption_threshold)
    
    def reset(cls):
        CountryAgent.instances = []
       # print('\n'.join(A.instances)) #this line was suggested by @anvelascos





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
        # Create agents
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
            #### based on Sebhatu et al. (2020) population dense countries are more likely to introduce lockdown
            #### by themselves
            a.alert_adoption_threshold = (agent_data["log_population_density_normalized_on_average"][i])**2 * (1/agent_data["democracy_index_normalized_on_average"][i]) * base_alert
            #### but the more democratic a country, the more sensitive they are to the influence of others 
            a.social_adoption_threshold = (agent_data["democracy_index_normalized_on_average"][i]) * social_base_threshold
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
                             "social_adoption_threshold": "social_adoption_threshold",
                             "alert_adoption_threshold": "alert_adoption_threshold",
                             "adoption_mode": "adoption_mode"
                             }
        )
    
    def step(self):
        
        self.time = self.time + 1
        
        
        if self.data_update == "yes" and i % 1 == 0:
             for agent in self.schedule.agents:
                                 agent.state = lockdown_data_for_calibration_micro[(agent.code==lockdown_data_for_calibration_micro.Code) & (self.time == lockdown_data_for_calibration_micro.model_step)]["lockdown"].iloc[0]
        else:
             pass

        self.datacollector.collect(self)
        self.schedule.step()
        

        
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
                   "politicalregime", "social_adoption_threshold", "alert_adoption_threshold",
                   "adoption_mode"]]
        df4.insert(0, "iteration", [j]*len(df4))
        
        if j == 0 :
            df_results = pd.DataFrame(columns = df4.columns)
            df_results = pd.concat([df_results, df4], join="inner")
        else:
            df_results = pd.concat([df_results, df4], join="inner", ignore_index=True)
        
        print("model iteration is " + str(j))
        
        CountryAgent.reset(CountryAgent)


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




fig1, ax1 = plt.subplots(figsize = (6,5))

for i in range(no_of_iterations):
    ax1.plot( df_results[(df_results.iteration == i) & (df_results.AgentID == 0)]["Step"] + 1,
              np.sum(np.split(np.array(df_results[(df_results.iteration == i)]["Lockdown"]),31),axis=1) / Num_agents * 100, alpha = 0.5)
ax1.plot(df_results[(df_results.iteration == i) & (df_results.AgentID == 0)]["Step"] + 1, 
         lockdown_data_for_calibration_agg[0]*100,
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

if no_of_iterations >= 10:
    
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

fig3, ax3 = plt.subplots(figsize=(12, 6))
for i in range(Num_agents):  
        model_lockdown_data = df_results[(df_results.code == agent_data["code"][i]) & (df_results.Lockdown == 1)]
        real_world_lockdown_date = lockdown_data_for_calibration_micro[(lockdown_data_for_calibration_micro.Code ==  agent_data["code"][i]) & (lockdown_data_for_calibration_micro.lockdown == 1)]
        if len(model_lockdown_data) > 0 and len(real_world_lockdown_date) > 0:
            difference = model_lockdown_data.iloc[0::31]["Step"] - (int(real_world_lockdown_date.iloc[0]["Day"][0:2])-1)
            ax3.scatter(np.repeat(i,len(difference)),difference, color = "tab:blue", alpha = 0.2)
        print("iteration of micro-level-plot is " + str(i))
ax3.set_ylabel("diff. in days")   
ax3.set_xlabel("country index")  
ax3.plot([0,164],[0,0], color = "black", linewidth = 3)
ax3.margins(0)
plt.savefig('Micro_validity_1.png', bbox_inches='tight', dpi=300)
plt.show()


### plot #4 fan chart of model runs 
##https://stackoverflow.com/questions/28807169/making-a-python-fan-chart-fan-plot
# THIS --> https://stackoverflow.com/questions/66146705/creating-a-fanchart-from-a-series-of-monte-carlo-projections-in-python


def create_fanchart(arr):
    x = np.arange(arr.shape[0])
    # for the median use `np.median` and change the legend below
    mean = np.mean(arr, axis=1)
    offsets = (25,67/2,47.5)
    fig, ax = plt.subplots()
    ax.plot(mean, color='black', lw=3)
    for offset in offsets:
        low = np.percentile(arr, 50-offset, axis=1)
        high = np.percentile(arr, 50+offset, axis=1)
        # since `offset` will never be bigger than 50, do 55-offset so that
        # even for the whole range of the graph the fanchart is visible
        alpha = (55 - offset) / 100
        ax.fill_between(x, low, high, color='tab:blue', alpha=alpha)
    
    ax.plot(df_results[(df_results.iteration == 0) & (df_results.AgentID == 0)]["Step"], 
             lockdown_data_for_calibration_agg[0]*100,
             linewidth=3 ,label = "data", linestyle= "--", color = "tab:red")
    
    ax.set_xlabel("Day of March")
    ax.set_ylabel("% of countries in lockdown")
    ax.legend(['Mean'] + [f'Pct{int(2*o)}' for o in offsets] + ['data'], frameon = False)
    return fig, ax

create_fanchart(array_run_results.T/Num_agents*100)
plt.savefig('fanchart_1_macro_validity.png', bbox_inches='tight', dpi=300)

#4.1 report least squares of mean to data and variance per time step 
## (both metrics need to minimized)
mean_model_runs = np.mean(array_run_results, axis = 0)
variance_model_runs = np.var(array_run_results, axis = 0)

### plot #5 micro validity over time as a function of how many countries are in their
### correct lockdown state (lockdown yes or no)

### test whether dataframes are in the exact same order
## test = df_results[(df_results.iteration == i) & (df_results.Step == j)]["code"] 
### == pd.Series.reset_index(lockdown_data_for_calibration_micro[(lockdown_data_for_calibration_micro.model_step == j)]["Code"], drop=True)

micro_validity_metric_array = np.zeros((31,no_of_iterations))
   
for i in range(no_of_iterations):
       micro_validity_metric_array[:,i] = np.mean(
                                                    np.array_split(
                                                                    np.array(
                                                                             pd.Series.reset_index(
                                                                                 df_results[(df_results.iteration == i) ]["Lockdown"], drop = True) 
                                                                             
                                                                             == 
                                                                                   
                                                                             pd.Series.reset_index(
                                                                                 lockdown_data_for_calibration_micro.sort_values(
                                                                                                   ['model_step', 'Entity']
                                                                                                   )
                                                                                           ["lockdown"],drop = True
                                                                                      )
                                                                           ),
                                                                31)
                                                 ,axis = 1
                                                )





squared_error_macro_level = (mean_model_runs/164 - lockdown_data_for_calibration_agg[0].to_numpy())**2
plt.scatter(np.log(np.var(micro_validity_metric_array, axis =1 )), np.log(squared_error_macro_level))
plt.xlabel("log(micro_validity_metric)")
plt.ylabel("log(squared_error_macro_level)")
plt.title("Predicting the accuracy of the model macro level from micro pattern")

plt.show()
plt.scatter(np.mean(micro_validity_metric_array, axis =1 ), squared_error_macro_level)
plt.show()



def create_fanchart_2(arr):
    x = np.arange(arr.shape[0])
    # for the median use `np.median` and change the legend below
    mean = np.mean(arr, axis=1)
    offsets = (25,67/2,47.5)
    fig, ax = plt.subplots()
    ax.plot(mean, color='black', lw=3)
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
    return fig, ax


create_fanchart_2(micro_validity_metric_array*100)
plt.savefig('fanchart_2_micro_validity.png', bbox_inches='tight', dpi=300)




#%%

non_adopters = df_results[(df_results.iteration == 0) & (df_results.Step == 30) & (df_results.Lockdown == 0)]



        
