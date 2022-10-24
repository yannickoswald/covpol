# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 14:40:49 2022

@author: earyo
"""


#%% Loading libraries

### import necessary libraries
import os
import mesa
import mesa.time
import mesa.space
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math as math
import pandas as pd
import copy as copy
from math import radians, cos, sin, asin, sqrt
import random
from datetime import datetime as datetime
import sys
# import random 
from random import sample
### colormaps import
import matplotlib.cm
##
import multiprocessing as mp
from multiprocessing import Pool
import csv

#work laptop path
os.chdir("C:/Users/earyo/Dropbox/Arbeit/postdoc_leeds/ABM_python_first_steps/implement_own_covid_policy_model")


#%%

data1 = pd.read_csv('N_of_particles_exp_without_pf.csv', 
                             encoding = 'unicode_escape',
                             header = 0, index_col=0)



data2 = pd.read_csv('N_of_particles_exp_with_pf.csv', 
                             encoding = 'unicode_escape',
                             header = 0, index_col=0)

powers_of_two = list(np.linspace(1,len(data1),len(data1)))
number_of_particles_per_experiment = [str(int(2**x)) for x in powers_of_two ]

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
ax2 = fig.add_axes([0.2, 0.1, 0.8, 0.8])
ax.boxplot(data1.T, widths = 0.25)

ax.set_xticklabels(number_of_particles_per_experiment)
ax.set_xlabel("number_of_particles")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)







ax2.boxplot(data2.T, widths = 0.25)
ax2.patch.set_alpha(0.01)
ax2.axis('off')





plt.scatter(2**num_power,sum(mse), label = "no filter", color = "tab:blue")
plt.scatter(2**num_power,sum(mse_pf), label = "particle filter", color = "tab:red")
plt.xscale("log", base=2)
plt.xlabel("Number of particles considered")
plt.ylabel("Sum of MSEs over time")  