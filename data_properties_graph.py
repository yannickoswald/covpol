# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 10:36:27 2022

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
### import model class (which itself imports agent class)
from model_class2 import CountryModel
from particle_filter_class import ParticleFilter

#work laptop path
os.chdir("C:/Users/earyo/Dropbox/Arbeit/postdoc_leeds/ABM_python_first_steps/implement_own_covid_policy_model")

#%%

data = pd.read_csv('correlation_between_policies_and_metrics.csv', encoding = 'unicode_escape')



#%%


fig1, (ax1,ax2) = plt.subplots(1,2, figsize = (10,4))

ax1.plot(np.linspace(1,31,31), data["level_0"])
ax1.plot(np.linspace(1,31,31), data["level_1"])
ax1.plot(np.linspace(1,31,31), data["level_2"])
ax1.plot(np.linspace(1,31,31), data["level_3"])
ax1.set_xlabel("Day in March")
ax1.set_ylabel("Number of countries")
ax1.set_ylim((0,175))




ax2.plot(np.linspace(1,31,31), data["school_closures"])
ax2.set_xlabel("Day in March")
ax2.set_ylim((0,175))







