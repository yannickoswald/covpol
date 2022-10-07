# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 08:09:46 2022

@author: earyo
"""

import matplotlib.pyplot as plt
import numpy as np


def create_fanchart(arr):
    x = np.arange(arr.shape[0])
    # for the median use `np.median` and change the legend below
    mean = np.mean(result, axis=1)
    offsets = (10, 20, 30, 40)
    fig, ax = plt.subplots()
    ax.plot(mean, color='black', lw=2)
    for offset in offsets:
        low = np.percentile(result, 50-offset, axis=1)
        high = np.percentile(result, 50+offset, axis=1)
        # since `offset` will never be bigger than 50, do 55-offset so that
        # even for the whole range of the graph the fanchart is visible
        alpha = (55 - offset) / 100
        ax.fill_between(x, low, high, color='blue', alpha=alpha)
    ax.legend(['Mean'] + [f'Pct{2*o}' for o in offsets])
    return fig, ax


S = 1
T = 180
mu = 0.15
vol = 0.05
samples = 100
result = []

for i in range(samples):
    monthly_returns = np.random.normal((1+mu)**(1/T), vol/np.sqrt(T), T)
    monthly_returns = np.hstack([1, monthly_returns])
    price_list = np.cumprod(monthly_returns) * S - 1
    result.append(price_list)
result = np.array(result).T

# plt.plot(result)  # for the individual samples
fig, ax = create_fanchart(result)
plt.show()