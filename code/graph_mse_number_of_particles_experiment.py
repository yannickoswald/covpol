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
# https://stackoverflow.com/questions/18386210/annotating-ranges-of-data-in-matplotlib


def draw_brace(ax, xspan, yy, text):
    """Draws an annotated brace on the axes."""
    xmin, xmax = xspan
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin

    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan/xax_span*100)*2+1  # guaranteed uneven
    beta = 300./xax_span  # the higher this is, the smaller the radius

    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:int(resolution/2)+1]
    y_half_brace = (1/(1.+np.exp(beta*(x_half-x_half[0])))
                    + 1/(1.+np.exp(beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = yy + (.05*y - .01)*yspan  # adjust vertical position

    ax.autoscale(False)
    ax.plot(x, y, color='black', lw=1)

    ax.text((xmax+xmin)/2., yy-.07*yspan, text, ha='center', va='bottom')

#%%


with open('C:/Users/earyo/Dropbox/Arbeit/postdoc_leeds/ABM_python_first_steps/implement_own_covid_policy_model/data/preliminary data on number of particles experiment aggregate.csv') as f:
    data0 = pd.read_csv(f, encoding='unicode_escape')


### select 32 particles to 512 particles for boxplot
### data2 is with particle filter
data2 = data0.iloc[4:11, :]

### data 1 is without particle filter
data1 = data0.iloc[15:22, :]

## select 1012 to 4096 particles for scatterplot next to boxplot
#data4 = data0.iloc[8:11]
#data3 = data0.iloc[19:22]

powers_of_two = list(np.linspace(2, len(data1)+1, len(data1)))
number_of_particles_per_experiment = [
    str(int(2**(x + 4))) for x in powers_of_two]
xticks = np.linspace(1, len(data1), len(data1))+0.25

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # main axes
ax2 = fig.add_axes([0.15, 0.1, 0.8, 0.8])

###https://stackoverflow.com/questions/41997493/python-matplotlib-boxplot-color
c = 'lightblue'
box1 = ax.boxplot(data1.T, widths=0.25, patch_artist=True,
                  boxprops=dict(facecolor=c, color=c),
                  medianprops=dict(color='black'))
ax.set_xticks(xticks)
ax.set_xticklabels(number_of_particles_per_experiment)
ax.set_xlabel("number of particles")
ax.set_ylabel("sum of MSE over time")
ax.set_xlim((0.5, len(data1)+1.1))
ax.set_ylim((0.1, 0.8))
ax.annotate("N=20 per boxplot", (3.5, 0.88), fontsize=12)
ax.legend()

for median in box1['medians']:
    median.set_color('tab:blue')


draw_brace(ax, (0.6, 4.7), 0.2, 'Iterations = 20')


draw_brace(ax, (4.9, 8), 0.2, 'Iterations = 1')


c2 = 'orange'
box2 = ax2.boxplot(data2.T, widths=0.25, patch_artist=True,
                   boxprops=dict(facecolor=c2, color=c2),
                   medianprops=dict(color='black'))
ax2.set_xticks(xticks)
ax2.patch.set_alpha(0.01)
ax2.set_ylim((0.1, 0.8))
ax2.axis('off')
ax2.set_xlim((0.5, len(data1)+1.1))

for median in box2['medians']:
    median.set_color('tab:red')

##https://stackoverflow.com/questions/47528955/adding-a-legend-to-a-matplotlib-boxplot-with-multiple-plots-on-same-axes
ax.legend([box1["boxes"][0], box2["boxes"][0]], [
          'NO PF', 'PF'], loc='upper right', frameon=False)

plt.savefig('MSE_number_of_particles_exp.png',  bbox_inches='tight', dpi=300)
