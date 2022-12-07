# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:40:45 2022

@author: earyo
"""


#%%
import os

from model_class2 import CountryModel

import pandas as pd
import numpy as np


#%% RUN THE MODEL
### run the model, without particle filter, and save data


#with Pool(processes=number_of_processors) as pool:     
#         data_results = list(pool.starmap(Experiment.run_experiment, inputs_for_starmap))

#start = dt.now()

class model_run():
        
    @classmethod
    def run_base_model_opt(cls, lockdown_data1, lockdown_data2):
        
                ##4th parameter initial conditions can be real, no countries yet or random
                model = CountryModel(0.01, 0.13, 18, 'real', 'no')
                for i in range(31):
                                
                                model.step()
                                df1 = model.datacollector.get_agent_vars_dataframe()
                                df2 = model.datacollector.get_model_vars_dataframe()
                                        
                ### here insert code for merging dataframes
                df3 = (df1.reset_index(level=0)).reset_index(level = 0)
                df4 = pd.DataFrame(np.repeat(df2.values, 200, axis=0))
                df4.columns = df2.columns
                df4 = pd.concat([df4, df3], axis=1, join='inner')
                df4 = df4[["code","AgentID", "Step", "minimum_difference", "Lockdown", "income",
                           "politicalregime", "social_thre", "own_thre",
                           "adoption_mode"]]
                #df4.insert(0, "iteration", [j]*len(df4))
                
                square_diffs = np.zeros((31,1))
                micro_validity_metric_array_small = np.zeros((31,1))
                   
                for i in range(31):
                    
                    model_macro_state = (df4[(df4.Step == i)]["Lockdown"].sum())/164
                    data_macro_state = lockdown_data1.iloc[i,:]
                    #make squared differences 
                    square_diffs[i,0] = (model_macro_state - data_macro_state)**2
                    
                    alpha = pd.Series.reset_index(df4[(df4.Step == i)]["Lockdown"],drop = True)
                    beta =  pd.Series.reset_index(lockdown_data2[(lockdown_data2.model_step == i)]["lockdown"],drop = True)         
                    micro_validity_metric_array_small[i,:] = np.mean(np.array(alpha == beta))
            
      
                return df4, square_diffs, micro_validity_metric_array_small