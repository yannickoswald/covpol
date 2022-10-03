# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 13:26:06 2022

@author: earyo
"""

#%%


from country_model_3 import *

#model = CountryModel(200, 0.02, 20, 10)
#for i in range(31):
 #   model.step()


#data = model.datacollector.get_agent_vars_dataframe()
#data2 = model.datacollector.get_model_vars_dataframe()


#%% 

### first initial plot showing the lockdowns over time ###

#data2.plot()


#%% batch run 


params = {"N": 200, "density": 0.04, "width": 20, "height": 10}

results = mesa.batch_run(
    CountryModel,
    parameters=params,
    iterations=10,
    max_steps=31,
    number_processes=1,
    data_collection_period=1,
    display_progress=True,
)


#%%

results_df = pd.DataFrame(results)
print(results_df.keys())


results_filtered = results_df[(results_df.AgentID == 150) & (results_df.Step == 31)]