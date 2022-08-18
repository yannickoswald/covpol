# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 14:27:46 2022

@author: earyo
"""


import mesa
import mesa.time
import mesa.space
import matplotlib.pyplot as plt
import numpy as np


def count_lockdowns(model):
    agent_lockdown_states = [agent.state for agent in model.schedule.agents]
    sum_lockdowns = sum(agent_lockdown_states)
    return sum_lockdowns




class CountryAgent(mesa.Agent):
    """Represents a single ALIVE or DEAD cell in the simulation."""
    
    
    NO_LOCKDOWN = 0
    LOCKDOWN = 1
    ### trying this based on this code 
    ### https://stackoverflow.com/questions/328851/printing-all-instances-of-a-class
    instances = []

        
    "initialize agent with income and democracy index only and drawn from a random distribution"
        
    def __init__(self, unique_id, model, init_state=NO_LOCKDOWN):
       super().__init__(unique_id, model)
       self.income = np.random.exponential(20000)
       self.politicalregime = np.random.normal(5.43, 2)
       self.adoption_threshold = np.random.normal (0.03, 0.01)
       self.minimum_difference = 0
       self.state = init_state
       self.__class__.instances.append(self)


    def compute_distance(self):
        
       if self.state == 0:
           
           ### find all countries that have implemented lockdown already
           ### and store in list
           ### also compute range of income
           max_income = 0
           min_income = 20000 
           
           list_of_lockdown_countries = []
           for i in range(0, len(CountryAgent.instances)):     
               if CountryAgent.instances[i].state == 1:
                   list_of_lockdown_countries.append(CountryAgent.instances[i])
               
               if CountryAgent.instances[i].income > max_income:
                       max_income = CountryAgent.instances[i].income
               if CountryAgent.instances[i].income < min_income:
                       min_income = CountryAgent.instances[i].income
                       
           range_income = max_income - min_income  
               
           #  compute distance function to all other countries that have lockdown
           ## income difference
           income_differences_array = np.zeros((len(list_of_lockdown_countries), 1)) 

           for i in range(0,len(income_differences_array)):
                income_differences_array[i,0] = self.income - list_of_lockdown_countries[i].income

           #income_differences_array = np.delete(income_differences_array, self.unique_id)
           self.minimum_difference = min(abs(income_differences_array)) / range_income
           
       else:
             pass 
                  
           
    def update_state(self):
           if self.minimum_difference < self.adoption_threshold:
               self.state = 1
               self.minimum_difference = 0 
                
    def step(self):
       self.compute_distance()
       self.update_state()

       # print('\n'.join(A.instances)) #this line was suggested by @anvelascos
       
       
       
       
       
       
       
                
class CountryModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N, density):
        self.num_agents = N
        self.schedule = mesa.time.RandomActivation(self)

        # Create agents
        for i in range(self.num_agents):
            a = CountryAgent(i, self)
            self.schedule.add(a)
            if self.random.random() < density:
                a.state = 1


        self.datacollector = mesa.DataCollector(
            model_reporters={"Number_of_lockdowns": count_lockdowns},
            agent_reporters={"minimum_difference": "minimum_difference", 
                             "Lockdown": "state"}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()




#%%


#from country_model import *

model = CountryModel(200, 0.05)
for i in range(31):
     model.step()


data = model.datacollector.get_agent_vars_dataframe()
data2 = model.datacollector.get_model_vars_dataframe()


#%% 

### first initial plot showing the lockdowns over time ###

data2.plot()

          