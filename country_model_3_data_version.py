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
#%%
### import necessary libraries
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

#%% READ DATA, DEFINE EXTERNAL FUNCTIONS
### read country/agent data
agent_data = pd.read_csv('agent_data_v1.csv', encoding = 'unicode_escape')
agent_data["gdp_pc"] = pd.to_numeric(agent_data["gdp_pc"])

lockdown_data_for_calibration_agg = pd.read_csv('lockdown_data_for_calibration.csv', 
                                                encoding = 'unicode_escape',
                                                header = None)
### this is a function that feeds into the model reporter and counts the lockdowns
### per time step for us

def count_lockdowns(model):
    agent_lockdown_states = [agent.state for agent in model.schedule.agents]
    sum_lockdowns = sum(agent_lockdown_states)
    return sum_lockdowns

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
       self.clique_size = 0
       self.state = init_state
       self.__class__.instances.append(self)
       if self.social_adoption_threshold <= 0.001:
              self.social_adoption_threshold  = 0.01
              
    # do something with the agent object
    def compute_distance(self):
        
       if self.state == 0:
           
           ### compute ranges of agent properties (later important for normalization
           ## of distance measure)
           max_income = max(agent_data["gdp_pc"]) ##
           min_income = min(agent_data["gdp_pc"])
           max_politicalregime = max(agent_data["democracy_index"])  ## ~ mean of the distr. as initial seed
           min_politicalregime = min(agent_data["democracy_index"])
           range_income = max_income - min_income
           range_politicalregime = max_politicalregime - min_politicalregime
           #range_covidcases = max_covidcases - min_covidcases
           ## max distance between two points on earth =
           ## earth circumference divided by two
           max_distance_on_earth = 40075.017/2
           
           
           ### find all countries that have implemented lockdown already
           ### and store in list
           list_of_lockdown_countries = []
           for i in range(0, len(CountryAgent.instances)):
               if CountryAgent.instances[i].state == 1:
                   list_of_lockdown_countries.append(CountryAgent.instances[i])
           
            #  compute distance function to all other countries that have lockdown
           income_differences_array = np.zeros(len(list_of_lockdown_countries))
           politicalregime_differences_array = np.zeros(len(list_of_lockdown_countries))
           #covidcases_differences_array = np.zeros((len(list_of_lockdown_countries), 1))
           geo_differences_array = np.zeros(len(list_of_lockdown_countries))
           total_differences_array  = np.zeros(len(list_of_lockdown_countries))
           
           for i in range(0,len(list_of_lockdown_countries)):
                 income_differences_array[i] = self.income - list_of_lockdown_countries[i].income
                 politicalregime_differences_array[i] = self.politicalregime - list_of_lockdown_countries[i].politicalregime
                 geo_differences_array[i] = geo_distance(self.latitude, list_of_lockdown_countries[i].latitude, 
                                                           self.longitude, list_of_lockdown_countries[i].longitude)
           
           total_differences_array  = ( 1/3 * (abs(income_differences_array)) / range_income
                                      + 1/3 * (abs(politicalregime_differences_array)) / range_politicalregime
                                      + 1/3 * (abs(geo_differences_array)) / max_distance_on_earth)
           
           #total_differences_array = np.delete(total_differences_array, np.where(total_differences_array == 0)[0][0])
           ### the division by max is a normalization on the interval [0,1]
           ### + 0.15 because the random-noise needs to be factored in the maximum range of possible values
           ### 0.15 is roughly calculated here as upper limited (3 std) to np.random.normal 0, 0.05
           ### https://onlinestatbook.com/2/calculators/normal_dist.html
           
           total_differences_array = np.sort(total_differences_array)
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
    
    def reset(cls):
        CountryAgent.instances = []
       # print('\n'.join(A.instances)) #this line was suggested by @anvelascos


class CountryModel(mesa.Model):
    
    """A model with some number of agents."""
    def __init__(self, base_alert, social_base_threshold, clique_size):
        self.num_agents = len(agent_data)
        self.base_alert = base_alert
        self.social_base_threshold = social_base_threshold
        self.schedule = mesa.time.RandomActivation(self)
        self.running = True
        self.clique_size = clique_size
        # Create agents
        for i in range(self.num_agents):
            a = CountryAgent(i, self)
            self.schedule.add(a)
            #if self.random.random() < density:
             #   a.state = 1
            a.state = agent_data["initial_state"][i]  
            a.name = agent_data["entity"][i]   
            a.income = agent_data["gdp_pc"][i]
            a.politicalregime = agent_data["democracy_index"][i]
            a.latitude = agent_data["capital_latitude"][i]
            a.longitude = agent_data["capital_longitude"][i]
            #### based on Sebhatu et al. (2020) population dense countries are more likely to introduce lockdown
            #### by themselves
            a.alert_adoption_threshold = (agent_data["log_population_density_normalized_on_average"][i])**2 * base_alert
            #### but the more democratic a country, the more sensitive they are to the influence of others 
            a.social_adoption_threshold = agent_data["democracy_index_normalized_on_average"][i] * social_base_threshold
            a.clique_size = self.clique_size
        
        self.datacollector = mesa.DataCollector(
            
            model_reporters={
                             ### lambda is anonymous function
                             "Number_of_lockdowns": count_lockdowns,
                             "N": lambda model: model.schedule.get_agent_count(),
                             "density": lambda density: self.base_alert,
                             },
            agent_reporters={
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
        self.datacollector.collect(self)
        self.schedule.step()
        
        
#%%
### run the model and save data

no_of_iterations = 10
for j in range(no_of_iterations):
        ### call the model iteration
        model = CountryModel(0.015, 0.12, 0.18)
        ## run each step
        for i in range(31):
                        model.step()
                        df1 = model.datacollector.get_agent_vars_dataframe()
                        df2 = model.datacollector.get_model_vars_dataframe()
       
        ### here insert code for merging dataframes
        df3 = (df1.reset_index(level=0)).reset_index(level = 0)
        df4 = pd.DataFrame(np.repeat(df2.values, 200, axis=0))
        df4.columns = df2.columns
        df4 = pd.concat([df4, df3], axis=1, join='inner')
        df4 = df4[["Number_of_lockdowns",
                   "AgentID", "Step", "minimum_difference", "Lockdown", "income",
                   "politicalregime", "social_adoption_threshold", "alert_adoption_threshold",
                   "adoption_mode"]]
        df4.insert(0, "iteration", [j]*len(df4))
        
        if j == 0 :
            df_results = pd.DataFrame(columns = df4.columns)
            df_results = pd.concat([df_results, df4], join="inner")
        else:
            df_results = pd.concat([df_results, df4], join="inner", ignore_index=True)
        
        print("iteration is " + str(j))
        
        CountryAgent.reset(CountryAgent)

### df_results_filtered
df_results_filtered = df_results[(df_results.AgentID == 9) & (df_results.Step == 30)]


#%%

#### PLOTTING

### Initial conditions 

#### plot #0.0 distributions of variables (income, democracy index, latitude and longitude)



### plot #0.1 map of lockdowns 





### plot #1 number of lockdowns over time steps
for i in range(no_of_iterations):
    plt.plot( df_results[(df_results.iteration == i) & (df_results.AgentID == 0)]["Step"] + 1,
              df_results[(df_results.iteration == i) & (df_results.AgentID == 0)]["Number_of_lockdowns"] / 172 * 100, alpha = 0.5)
plt.plot(df_results[(df_results.iteration == i) & (df_results.AgentID == 0)]["Step"] + 1, 
         lockdown_data_for_calibration_agg[0]*100,
         linewidth=3 ,label = "data")
plt.xlabel("Day of March 2020")
plt.ylabel("% of countries in lockdown")
plt.legend(frameon=False)
plt.text(23, 20, "Clique size: " + str(model.clique_size))
plt.text(23, 15, "Base alert: " + str(model.base_alert))
plt.text(23, 10, "Social alert: " + str(model.social_base_threshold))
plt.show()

### plot #2 average minimum_difference (should decay over time)
### because more countries adopt a lockdown so for each country more and more similar countries 
### serve as a benchmark

for i in range(no_of_iterations):
    df_iter = df_results[(df_results.iteration == i)]
    average_min_diff_array = np.zeros((31,1))
    average_min_and_max_diff = np.zeros((31,2))
    for j in range(31):
           average_min_diff_array[j,0] = np.mean( df_iter[(df_iter.Step == j)]["minimum_difference"])
           average_min_and_max_diff[j,0] = min(df_iter[(df_iter.Step == j)]["minimum_difference"])
           average_min_and_max_diff[j,1] = max(df_iter[(df_iter.Step == j)]["minimum_difference"])
                         
    plt.plot((df_results[(df_results.iteration == i) & (df_results.AgentID == 0)]["Step"] + 1)[1:30], average_min_diff_array[1:30])
    #plt.plot((df_results[(df_results.iteration == i) & (df_results.AgentID == 0)]["Step"] + 1)[1:30], average_min_and_max_diff[1:30,0])
    #plt.plot((df_results[(df_results.iteration == i) & (df_results.AgentID == 0)]["Step"] + 1)[1:30], average_min_and_max_diff[1:30,1])
       
plt.xlabel("Day of March 2020")
plt.ylabel("Average min distance")
plt.show()

### plot #3....

#%%
