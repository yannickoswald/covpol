# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 12:59:54 2022

@author: earyo
"""

particle = list_of_particles_filtered[10]

advance_particle(particle)


t = particle.time
### go through all datasteps necessary to find error
data1 = copy.deepcopy(particle.datacollector.get_agent_vars_dataframe())
data2 = (data1.reset_index(level=0)).reset_index(level = 0)
data3 = data2[(data2.Step == t)]
data4 = pd.Series.reset_index(data3["Lockdown"], drop = True)
data5 = lockdown_data2[(lockdown_data2.model_step == t)]["lockdown"]
data6 = pd.Series.reset_index(data5, drop = True)
particle_validity = np.mean(data4 == data6)

error_particle_obs(particle)
