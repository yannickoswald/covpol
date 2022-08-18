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

class Country(mesa.Agent):
    """Represents a single ALIVE or DEAD cell in the simulation."""

    NO_LOCKDOWN = 0
    LOCKDOWN = 1
    ### trying this based on this code 
    ### https://stackoverflow.com/questions/328851/printing-all-instances-of-a-class
    instances = []

        
    "initialize agent with income and democracy index only and drawn from a random distribution"
        
    def __init__(self, unique_id, model):
       super().__init__(unique_id, model)
       self.income = np.random.exponential(5e-05)
       self.democracy_index = np.random.normal(5.43, 2)
       self.adoption_threshold = np.random.normal (0.2, 0.01)
       self.minimum_difference = 0
       self.__class__.instances.append(self)


    def compute_distance(self):
  
     #   compute distance function to all other countries
       income_differences_array = np.zeros((len(instances), 1)) 
       
       for i in range(0,len(instances)):
            income_differences_array[i,0] = self.income - Country.instances[i].income
       self.minimum_difference = min(income_differences_array)
                

       # print('\n'.join(A.instances)) #this line was suggested by @anvelascos
       
       
       
       
       
       
                
class CountryModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N, width, height):
        self.num_agents = N
        self.schedule = mesa.time.RandomActivation(self)

        # Create agents
        for i in range(self.num_agents):
            a = MoneyAgent(i, self)
            self.schedule.add(a)


        self.datacollector = mesa.DataCollector(
            agent_reporters={"Minimum_difference": "minimum_difference"}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()




#%%


from money_model import *

model = MoneyModel(50, 10, 10)
for i in range(100):
    model.step()


gini = model.datacollector.get_model_vars_dataframe()
gini.plot()
# If running from a text editor or IDE, remember you'll need the following:
plt.show()
