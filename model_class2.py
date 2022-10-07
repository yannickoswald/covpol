# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 10:39:42 2022

@author: earyo
"""



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
import csv

#work laptop path
os.chdir("C:/Users/earyo/Dropbox/Arbeit/postdoc_leeds/ABM_python first steps/implement own covid policy model")


#%% import agents

from agent_class1 import CountryAgent

#%%

class CountryModel(mesa.Model):
    
    """A model with some number of agents."""
    
    def __init__(self, base_alert, social_base_threshold, 
                 clique_size, initial_conditions, data_update):
        self.num_agents = 164
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
        
    def instantiate_agents_from_csv(self):

        ## open csv files with data and storeas pandas dataframe
        agent_data = pd.read_csv('agent_data_v2.csv', encoding = 'unicode_escape')
        agent_data["gdp_pc"] = pd.to_numeric(agent_data["gdp_pc"])

        ### loop that creates all agents and sets the agent attributes
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
            z = self.base_alert
            a.own_thre = (x**2) * (1/y) * z
            #### but the more democratic a country,
            #### the more sensitive they are to the influence of others 
            x1 = agent_data["democracy_index_normalized_on_average"][i]
            y1 = self.social_base_threshold
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
        

    def __repr__(self):
            return f"{self.__class__.__name__}('Model_id {self.model_id}')"
    
    def step(self):
        
        self.time = self.time + 1
        self.datacollector.collect(self)
        self.schedule.step()
        
