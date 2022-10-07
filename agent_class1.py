# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 10:35:04 2022

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
         
    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}')"
    
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