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

#%%
### read country/agent data
agent_data = pd.read_csv('agent_data_v1.csv', encoding = 'unicode_escape')  


#%%



def count_lockdowns(model):
    agent_lockdown_states = [agent.state for agent in model.schedule.agents]
    sum_lockdowns = sum(agent_lockdown_states)
    return sum_lockdowns

    

class CountryAgent(mesa.Agent):
    """Represents a single ALIVE or DEAD cell in the simulation."""
    
    
    NO_LOCKDOWN = 0
    LOCKDOWN = 1
    income_begin = 0
    politicalregime_begin = 0
    latitude_begin = 0
    longitude_begin = 0
    ### trying this based on this code 
    ### https://stackoverflow.com/questions/328851/printing-all-instances-of-a-class
    instances = []

        
    "initialize agent with income and democracy index"
        
    def __init__(self, unique_id, model, init_income = income_begin,
                 init_politicalregime = politicalregime_begin, 
                 init_latitude = latitude_begin, 
                 init_longitude = longitude_begin, 
                 init_state=NO_LOCKDOWN):
        
       super().__init__(unique_id, model)
       self.income = init_income
       self.politicalregime = init_politicalregime
       self.latitude = init_latitude
       self.longitude = init_longitude
       self.covidcases = np.random.gamma(0.3,1)
       self.adoption_threshold = np.random.normal (0.15, 0.15 * 0.2)
       self.minimum_difference = 1
       self.state = init_state
       self.__class__.instances.append(self)
       if self.adoption_threshold <= 0.001:
              self.adoption_threshold  = 0.01 


    
    # do something with the agent object


    def compute_distance(self):
        
       if self.state == 0:
           
           ### find all countries that have implemented lockdown already
           ### and store in list
           ### also compute ranges of agent properties
            
           max_income = 20000 ## ~ mean of the distr. as initial seed
           min_income = 0 
           
           max_politicalregime = 5.43  ## ~ mean of the distr. as initial seed
           min_politicalregime = 0 
           
           max_covidcases = 0.4 ## ~ mean of the distr. as initial seed
           min_covidcases = 0
           

           list_of_lockdown_countries = []
           
           
           for i in range(0, len(CountryAgent.instances)):     
               if CountryAgent.instances[i].state == 1:
                   list_of_lockdown_countries.append(CountryAgent.instances[i])
               
                
               ## find min and max income
               if CountryAgent.instances[i].income > max_income:
                       max_income = copy.deepcopy(CountryAgent.instances[i].income)
               if CountryAgent.instances[i].income < min_income:
                       min_income = copy.deepcopy(CountryAgent.instances[i].income)
                       
               ## find min and max politicalregime        
               if CountryAgent.instances[i].politicalregime > max_politicalregime:
                      max_politicalregime = copy.deepcopy(CountryAgent.instances[i].politicalregime)
               if CountryAgent.instances[i].income < min_politicalregime:
                      min_politicalregime = copy.deepcopy(CountryAgent.instances[i].politicalregime)
                        
               ## find min and max covidcases       
               if CountryAgent.instances[i].covidcases > max_covidcases:
                      max_covidcases = copy.deepcopy(CountryAgent.instances[i].covidcases)
               if CountryAgent.instances[i].covidcases < min_covidcases:
                      min_covidcases = copy.deepcopy(CountryAgent.instances[i].covidcases)
                       
           range_income = max_income - min_income
           range_politicalregime = max_politicalregime - min_politicalregime
           range_covidcases = max_covidcases - min_covidcases
               
           #  compute distance function to all other countries that have lockdown
           
           income_differences_array = np.zeros((len(list_of_lockdown_countries), 1))
           politicalregime_differences_array = np.zeros((len(list_of_lockdown_countries), 1))
           covidcases_differences_array = np.zeros((len(list_of_lockdown_countries), 1))
           geo_differences_array = np.zeros((len(list_of_lockdown_countries), 1))
           total_differences_array  = np.zeros((len(list_of_lockdown_countries), 1))
    
           for i in range(0,len(list_of_lockdown_countries)):
                       
                 income_differences_array[i,0] = self.income - list_of_lockdown_countries[i].income
                 politicalregime_differences_array[i,0] = self.politicalregime - list_of_lockdown_countries[i].politicalregime
                 covidcases_differences_array[i,0] = self.covidcases - list_of_lockdown_countries[i].covidcases
                 geo_differences_array[i,0] = np.sqrt((self.pos[0] - list_of_lockdown_countries[i].pos[0]) ** 2 + (self.pos[1] - list_of_lockdown_countries[i].pos[1]) ** 2)
        
           total_differences_array  = ((abs(income_differences_array)) / range_income
                                              + (abs(politicalregime_differences_array)) / range_politicalregime
                                              + (abs(covidcases_differences_array)) / range_covidcases
                                              + (abs(geo_differences_array)) / np.sqrt( 20 * 20 + 10 * 10 ))
                                              
           #total_differences_array = np.delete(total_differences_array, np.where(total_differences_array == 0)[0][0])
                   
                   
           self.minimum_difference = ((min(total_differences_array, default=0)
                                               + np.random.normal(0,0.05)) 
                                              / (max(total_differences_array, default = 0) + 0.15))
                   
           ### the division by max is a normalization on the interval [0,1]
           ### + 0.15 because the random-noise needs to be factored in the maximum range of possible values
           ### 0.15 is roughly calculated here as upper limited (3 std) to np.random.normal 0, 0.05 
           ### https://onlinestatbook.com/2/calculators/normal_dist.html
           
           list_of_lockdown_countries.clear()
           
       else:
             pass 
                  
           
    def update_state(self):  
           self.covidcases = self.covidcases + np.random.normal (0, 0.02)
           if self.minimum_difference < self.adoption_threshold:
               self.state = 1
     
                
    def step(self):
       self.compute_distance()
       self.update_state() 
       
    def reset(cls):
        CountryAgent.instances = []

       # print('\n'.join(A.instances)) #this line was suggested by @anvelascos
       
       
       
      
        
      
                
class CountryModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, density):
        self.num_agents = len(agent_data)
        self.density = density
        self.schedule = mesa.time.RandomActivation(self)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            a = CountryAgent(i, self)
            self.schedule.add(a)
            if self.random.random() < density:
                a.state = 1
            a.income = agent_data["gdp_pc"][i]
            a.politicalregime = agent_data["democracy_index"][i]
            a.latitude = agent_data["capital_latitude"][i]
            a.longitude = agent_data["capital_longitude"][i]
            


        ### lambda is anonymous function
        self.datacollector = mesa.DataCollector(
            
            model_reporters={
                             "Number_of_lockdowns": count_lockdowns,
                             "N": lambda model: model.schedule.get_agent_count(),
                             "density": lambda density: self.density,
                             "width": lambda density: self.width,
                             "height": lambda density: self.height
                             },
            
            agent_reporters={
                             "minimum_difference": "minimum_difference", 
                             "Lockdown": "state",
                             "income": "income",
                             "politicalregime": "politicalregime",
                             "covidcases": "covidcases",
                             "adoption_threshold": "adoption_threshold"
                             }
        )
             

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
    
    
   




#%%
### run the model and save data

no_of_iterations = 100


for j in range(no_of_iterations):
    
        ### call the model iteration
        model = CountryModel(0.05)
        
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

        df4 = df4[["density", "Number_of_lockdowns",
                   "AgentID", "Step", "minimum_difference", "Lockdown", "income",
                   "politicalregime", "covidcases", "adoption_threshold"]]   

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
        
#data2.plot()
#print(data2.iloc[30])



#### covidcases become negative change this 
#### insert column for number of iterations


#%%
