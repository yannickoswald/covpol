# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 12:43:46 2022

@author: earyo
"""

#%%
class model_constrainer_naive(constrainer):
    
    def __init__(self, model, data, update_period, iterations):
        self.model = model
        self.data = data
        self.update_period = update_period
        self.iterations = iterations
        
    def run_model_with_updates(self):
                for j in range(no_of_iterations):
                        ### call the model iteration
                        model = CountryModel(0.015, 0.12, 18, 'real')
                        for i in range(31):
                                        model.step()
                                        df1 = model.datacollector.get_agent_vars_dataframe()
                                        df2 = model.datacollector.get_model_vars_dataframe()
                       
                        ### here insert code for merging dataframes
                        df3 = (df1.reset_index(level=0)).reset_index(level = 0)
                        df4 = pd.DataFrame(np.repeat(df2.values, 200, axis=0))
                        df4.columns = df2.columns
                        df4 = pd.concat([df4, df3], axis=1, join='inner')
                        df4 = df4[["Number_of_lockdowns",
                                   "code","AgentID", "Step", "minimum_difference", "Lockdown", "income",
                                   "politicalregime", "social_adoption_threshold", "alert_adoption_threshold",
                                   "adoption_mode"]]
                        df4.insert(0, "iteration", [j]*len(df4))
                        
                        if j == 0 :
                            df_results = pd.DataFrame(columns = df4.columns)
                            df_results = pd.concat([df_results, df4], join="inner")
                        else:
                            df_results = pd.concat([df_results, df4], join="inner", ignore_index=True)
                        
                        print("model iteration is " + str(j))
                        
                        CountryAgent.reset(CountryAgent)
        