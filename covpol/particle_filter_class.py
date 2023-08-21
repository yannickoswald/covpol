# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 10:12:36 2022

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

### for explanation of pathos multiprocessing
#### https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function
#### https://edbennett.github.io/high-performance-python/08-pathos/index.html
#from pathos.pools import ParallelPool
from multiprocessing import Pool

##
import pytest

#######################
#### IMPORT MODEL #####
#######################

from model_class import CountryModel



#### data per country
with open('../data/lockdown_tracking.csv') as f:
    lockdown_data2  = pd.read_csv(f, encoding = 'unicode_escape')


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
        - No_of_particles:         The number of particles used to simulate the model
        - da_window:               The number of days between filtering steps
        - da_instances:            The number of times that filtering is undertaken

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
       
       
       #self.pool = ParallelPool(processes=8)
       
    
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
                list_of_particles.append(current_model) 
                
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
        ### here compare ultimately model microstate (data4) with obs. (data6)
        particle_validity = np.mean(data4 == data6)
        return particle_validity
    
    @classmethod
    def particle_weights(cls, error_list):
        
        '''DESCRIPTION
            ### Method that computes all particle weights from particle error/validity metric.
            ### Taken "Squared" to "penalize" low ranking particles more.
            ### Weight equals error squared because error close to 0 means high 
            ### error, so weight should also be low if error is close to 0.
        
           PARAMETERS
           - error list:        the errors of the particles  '''

        weights = [error**2 for error in error_list]
        ### normalization constraint such that sum(weights) = 1
        weights = weights/sum(weights)    
        return weights
    
    
    @classmethod
    def resample_particles(cls, list_of_particles, weights):
        
        '''DESCRIPTION
            Method that resamples the particles after each data assimilation
            window. The function works based on sequential importance resampling,
            where every particle weight determines its likelihood to be 
            resampled. The weights are cumulatively counted, so they constitute a
            cumulative distr. function (CDF) -- a weight distribution. And this distr. is
            compared against a uniformly random partition of the interval [0,1]
            constituting a uniformly random CDF. If the uniformly random CDF 'makes' 
            larger steps than the weights cumulation, because the weights are small, 
            it is likely that particles are filtered out.
        
           PARAMETERS
           - list_of_particles_arg:        the errors of the particles 
           - weights_arg:                  list of weights     '''
        
    
        
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
                        
        #### IMPORTANT: here deepcopy is necessary otherwise it creates
        #### reference between resampled particles with same model_id
        #### and messes up the model   
             
        list_of_particles_new = [copy.deepcopy(list_of_particles[int(x)])
                                 for x in re_sampled_particles]
        return list_of_particles_new
    
    @classmethod
    def step_particle(cls, particle):  
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

        print(list_of_particles)
        ### if parallized then use with Pool() as pool:
        ### i here for time steps in the model, the month of March 2020
        for i in range(31):          
                 
                    ## step particles forward in time
                    for x in list_of_particles:
                        ParticleFilter.step_particle(x)
        
                    #list_of_particles = list(pool.map(ParticleFilter.step_particle, list_of_particles))
                        
                    ## go into the data assimilitation if this time step is actually
                    ## at the end of a data assimilation window
                    if (i > 0) and (i % self.da_window == 0):
                        
                        list_of_errors = [ParticleFilter.error_particle_obs(k) 
                                          for k in list_of_particles]
                
                        weights = ParticleFilter.particle_weights(list_of_errors)
                        list_of_particles = ParticleFilter.resample_particles(list_of_particles,
                                                                              weights)
                        list_of_lists_weights.append(weights)
                        
                    list_of_lists_particles.append(list_of_particles)   
                    
                    print("Particle filter is at time step ", i)
    
        self.part_filtered_all = list_of_lists_particles
        self.weights = list_of_lists_weights