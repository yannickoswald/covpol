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

powers_of_two = list(np.linspace(2,len(data1)+1,len(data1)))
number_of_particles_per_experiment = [str(int(2**x)) for x in powers_of_two ]
xticks = np.linspace(1,len(data1),len(data1))+0.25

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
ax2 = fig.add_axes([0.15, 0.1, 0.8, 0.8])

###https://stackoverflow.com/questions/41997493/python-matplotlib-boxplot-color
c = 'lightblue'
box1 = ax.boxplot(data1.T, widths = 0.25, patch_artist=True,
                  boxprops=dict(facecolor=c, color=c),
                  medianprops=dict(color='black'))
ax.set_xticks(xticks)
ax.set_xticklabels(number_of_particles_per_experiment)
ax.set_xlabel("number of particles")
ax.set_ylabel("sum of MSE over time")
ax.set_xlim((0.5,len(data1)+1.1))
ax.annotate("N=20 per boxplot", (3.5,0.88), fontsize = 12)
ax.legend()



c2 = 'orangered'
box2 = ax2.boxplot(data2.T, widths = 0.25, patch_artist=True,
                   boxprops=dict(facecolor=c2, color=c2),
                   medianprops=dict(color='black'))
ax2.set_xticks(xticks)
ax2.patch.set_alpha(0.01)
ax2.axis('off')
ax2.set_xlim((0.5,len(data1)+1.1))


##https://stackoverflow.com/questions/47528955/adding-a-legend-to-a-matplotlib-boxplot-with-multiple-plots-on-same-axes
ax.legend([box1["boxes"][0], box2["boxes"][0]], ['NO PF', 'PF'], loc='upper right', frameon = False)

plt.savefig('MSE_number_of_particles_exp.png',  bbox_inches='tight', dpi=300)


