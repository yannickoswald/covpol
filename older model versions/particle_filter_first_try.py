# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 08:59:44 2022

@author: earyo
"""

#%%




 
# Driver program
arr = np.expand_dims(micro_validity_metric_array[10,:],1)
#arr = np.expand_dims(np.array([23, 12, 1, 9, 30, 2, 50, 11, 14, 15]),1)
arr_index = np.expand_dims(np.linspace(0,len(arr)-1,len(arr)),1)
arr_out = np.concatenate((arr, arr_index), axis = 1)
arr_out2 = arr_out[np.argsort(arr_out [:, 0])]
indices_out = arr_out2[0:n_of_filterings_per_step,1]


find_minima_index(micro_validity_metric_array[10,:])



####Particle Filter 1st try 


# parameters
## N = number of particles
## daw = data_assimilation_window


### advance simulation of particles to time = k, then halt
k = 10 

### create and store particles 

No_of_particles = 5
list_of_particles = []
list_of_weights = []
for i in range(No_of_particles):
        ### call the model iteration
        ##4th parameter initial conditions can be real, no countries yet or random
        current_model = CountryModel(0.01, 0.13, 18, 'real', 'no') 
        list_of_particles.append(current_model) 
        
for i in range(No_of_particles): 
        for j in range(k):             
                        list_of_particles[i].step()
        
             
 
### compute error metric for each particle

#for i in range(No_of_particles):



def error_particle_obs(t, particle):
    
    ### parameters 
    ### t for time or step i.e. at which time step is the error calculated
    ### particle because a specific particle needs to be passed    

    data1 = particle.datacollector.get_agent_vars_dataframe()
    data2 = (data1.reset_index(level=0)).reset_index(level = 0)    
    fraction_correctly_estimated = np.mean(pd.Series.reset_index(data2[(data2.Step == (t-1))]["Lockdown"], drop = True) == pd.Series.reset_index(lockdown_data_for_calibration_micro[(lockdown_data_for_calibration_micro.model_step == (t-1))]["lockdown"], drop = True))
    return 1 - fraction_correctly_estimated                   
                                                   
for i in range(No_of_particles):
    
    data1 = list_of_particles[i].datacollector.get_agent_vars_dataframe()
    data2 = (data1.reset_index(level=0)).reset_index(level = 0)    
    fraction_correctly_estimated = np.mean(pd.Series.reset_index(data2[(data2.Step == (t-1))]["Lockdown"], drop = True) == pd.Series.reset_index(lockdown_data_for_calibration_micro[(lockdown_data_for_calibration_micro.model_step == (t-1))]["lockdown"], drop = True))
    
    list_of_weights.append(error_particle_obs(k, list_of_particles[i]))
   
   


def resample():
    
### parameters
### particle_population


##### delete m particles with smallest weights



#### resample based on uniform draw from other population





### continue simulation of resampled particle population after data assimilation step is finished










            df1 = model.datacollector.get_agent_vars_dataframe()
                df2 = model.datacollector.get_model_vars_dataframe()
                
          

### here insert code for merging dataframes
df3 = (df1.reset_index(level=0)).reset_index(level = 0)
df4 = pd.DataFrame(np.repeat(df2.values, 200, axis=0))
df4.columns = df2.columns
df4 = pd.concat([df4, df3], axis=1, join='inner')
df4 = df4[["code","AgentID", "Step", "minimum_difference", "Lockdown", "income",
           "politicalregime", "social_adoption_threshold", "alert_adoption_threshold",
           "adoption_mode"]]
df4.insert(0, "iteration", [j]*len(df4))

if j == 0 :
    df_results = pd.DataFrame(columns = df4.columns)
    df_results = pd.concat([df_results, df4], join="inner")
else:
    df_results = pd.concat([df_results, df4], join="inner", ignore_index=True)
    
    
    
    
    
    
    
#%%


#%%

## Fake? (ex-post or very naive and vanilla) Particle Filter for conference example only

### GLOBAL PF parameters
percentage_filtered = 10
n_of_filterings_per_step = int(no_of_iterations * percentage_filter/100)
da_window = 10
da_instances = int(30/da_window)

def find_minima_index(arr):
    
        #### function to find the indices of the k minima in an array
        ### used to find least to data fitting particles
        
        arr = np.expand_dims(arr,1)
        #arr = np.expand_dims(np.array([23, 12, 1, 9, 30, 2, 50, 11, 14, 15]),1)
        arr_index = np.expand_dims(np.linspace(0,len(arr)-1,len(arr)),1)
        arr_out = np.concatenate((arr, arr_index), axis = 1)
        arr_out2 = arr_out[np.argsort(arr_out [:, 0])]
        indices_out = arr_out2[0:n_of_filterings_per_step,1]
        return indices_out

#micro_validity_metric_array[10,:]
df_results_particle_filtered = copy.deepcopy(df_results)

for k in range(1, da_instances):
    
    ### determine which runs need to be filtered out
    to_be_filtered = find_minima_index(micro_validity_metric_array[k*da_window,:]).astype(int)    

    #to_be_filtered = find_minima_index(micro_validity_metric_array[10,:]).astype(int)
    ### prepare a copy of actual results which will serve as filtered results storage
    
    ### code that finds the relevant particles, which are not thrown out, but from which 
    ### resampling is conducted. the indices of these particles are found
    sample_from_indices_arr = np.linspace(0,len(micro_validity_metric_array[10,:])-1,
                                          len(micro_validity_metric_array[10,:]))
    sample_from_indices_arr = np.delete(sample_from_indices_arr, to_be_filtered)
    
    #### code that creates a random sample of particles (from a uniform distr. over the particles that are
    ### kept )
    sample_arr = np.zeros((n_of_filterings_per_step,1))
    for i in range(n_of_filterings_per_step):
       sample_arr[i,0] = sample(sample_from_indices_arr.tolist(),1)[0]
       
    ### code that replaces the particles that need to be filtered out, with the resample particles 
    for i in range(len(to_be_filtered)):
        df_results_particle_filtered.loc[df_results_particle_filtered['iteration'] == to_be_filtered[i] , 'Lockdown'] = df_results_particle_filtered[(df_results_particle_filtered.iteration == sample_arr[i][0])]["Lockdown"].to_numpy()
        df_results_particle_filtered.loc[df_results_particle_filtered['iteration'] == to_be_filtered[i] , 'minimum_difference'] = df_results_particle_filtered[(df_results_particle_filtered.iteration == sample_arr[i][0])]["minimum_difference"].to_numpy()
        df_results_particle_filtered.loc[df_results_particle_filtered['iteration'] == to_be_filtered[i] , 'adoption_mode'] = df_results_particle_filtered[(df_results_particle_filtered.iteration == sample_arr[i][0])]["adoption_mode"].to_numpy()
        
    
    
array_run_results_particle_filtered = np.zeros((no_of_iterations,31))
for i in range(no_of_iterations):    
    array_run_results_particle_filtered[i,:] = np.sum(np.split(np.array(df_results_particle_filtered[(df_results_particle_filtered.iteration == i)]["Lockdown"]),31),axis=1)




def create_fanchart(arr):
    x = np.arange(arr.shape[0])+1
    # for the median use `np.median` and change the legend below
    mean = np.mean(arr, axis=1)
    offsets = (25,67/2,47.5)
    fig, ax = plt.subplots()
    ax.plot(x, mean, color='black', lw=3)
    ax.set_ylim([0,100])
    for offset in offsets:
        low = np.percentile(arr, 50-offset, axis=1)
        high = np.percentile(arr, 50+offset, axis=1)
        # since `offset` will never be bigger than 50, do 55-offset so that
        # even for the whole range of the graph the fanchart is visible
        alpha = (55 - offset) / 100
        ax.fill_between(x, low, high, color='tab:blue', alpha=alpha)
    
    ax.plot(df_results_particle_filtered[(df_results_particle_filtered.iteration == 0) & (df_results_particle_filtered.AgentID == 0)]["Step"] + 1, 
             lockdown_data_for_calibration_agg[0]*100,
             linewidth=3 ,label = "data", linestyle= "--", color = "tab:red")
    
    ax.set_xlabel("Day of March")
    ax.set_ylabel("% of countries in lockdown")
    ax.legend(['Mean'] + [f'Pct{int(2*o)}' for o in offsets] + ['data'], frameon = False)
    for i in range(1,da_instances):
        ax.plot([i*da_window, i*da_window],[0,110], color='lightgrey', lw=2 , alpha = 0.8)
    ax.margins(x=0)
    return fig, ax

create_fanchart(array_run_results_particle_filtered.T/Num_agents*100)
plt.savefig('fanchart_1_macro_validity_naive_pf.png', bbox_inches='tight', dpi=300)
plt.show()



micro_validity_metric_array_particle_filtered = np.zeros((31,no_of_iterations))
   
for i in range(no_of_iterations):
    micro_validity_metric_array_particle_filtered[:,i] = np.mean(
                                                    np.array_split(
                                                                    np.array(
                                                                             pd.Series.reset_index(
                                                                                 df_results_particle_filtered[(df_results_particle_filtered.iteration == i) ]["Lockdown"], drop = True) 
                                                                             
                                                                             == 
                                                                                   
                                                                             pd.Series.reset_index(
                                                                                 lockdown_data_for_calibration_micro.sort_values(
                                                                                                   ['model_step', 'Entity']
                                                                                                   )
                                                                                           ["lockdown"],drop = True
                                                                                      )
                                                                           ),
                                                                31)
                                                 ,axis = 1
                                                )





def create_fanchart_2(arr):
    x = np.arange(arr.shape[0])+1
    # for the median use `np.median` and change the legend below
    mean = np.mean(arr, axis=1)
    offsets = (25,67/2,47.5)
    fig, ax = plt.subplots()
    ax.plot(x, mean, color='black', lw=3)
    for offset in offsets:
        low = np.percentile(arr, 50-offset, axis=1)
        high = np.percentile(arr, 50+offset, axis=1)
        # since `offset` will never be bigger than 50, do 55-offset so that
        # even for the whole range of the graph the fanchart is visible
        alpha = (55 - offset) / 100
        ax.fill_between(x, low, high, color='tab:red', alpha=alpha)
        ax.set_xlabel("Day of March")
        ax.set_ylabel("% of countries in correct state")
        ax.legend(['Mean'] + [f'Pct{int(2*o)}' for o in offsets] + ['data'], frameon = False)
    ax.margins(x=0)
    return fig, ax


create_fanchart_2(micro_validity_metric_array_particle_filtered*100)
plt.savefig('fanchart_2_micro_validity_naive_pf.png', bbox_inches='tight', dpi=300)



##### PF EXPERIMENTS METRICS (SIMPLE VERSION)
mean_model_runs_naive_pf = np.mean(array_run_results_particle_filtered, axis = 0)

mean_at_t_15_pf = mean_model_runs_naive_pf[15]/164
datapoints_at_t_15_pf = array_run_results_particle_filtered[:,15]/164


####


mean_model_runs_naive_pf = np.mean(array_run_results_particle_filtered, axis = 0)
squared_error_macro_level_naive_pf = ((mean_model_runs_naive_pf/164 - lockdown_data_for_calibration_agg[0].to_numpy())**(2))*100
mean_squared_error_macro_level_naive_pf = (1/31)*(mean_model_runs_naive_pf/164 - lockdown_data_for_calibration_agg[0].to_numpy())**2
variance_model_runs_particle_filtered = np.var(array_run_results_particle_filtered, axis = 0)
mean_variance_model_runs_particle_filtered = np.var(array_run_results_particle_filtered, axis = 0)  




number_of_particles = 10
re_sampled_particles = np.zeros(number_of_particles)
random_partition_one_to_zero = ((np.arange(number_of_particles)
                             + np.random.uniform()) / number_of_particles)
cumsum = np.cumsum(weights)
i, j = 0, 0
while i < number_of_particles :
            if random_partition_one_to_zero[i] < cumsum[j]:
                re_sampled_particles[i] = j
                i += 1
            else:
                j += 1
                


        self.states[:] = self.states[self.indexes]
        self.weights[:] = self.weights[self.indexes]
        

 def resample(self):
        '''
        Resample
        DESCRIPTION
        Calculate a random partition of (0,1) and then
        take the cumulative sum of the particle weights.
        Carry out a systematic resample of particles.
        Set the new particle states and weights and then
        update agent locations in particle models using
        multiprocessing methods.
        '''
        offset_partition = ((np.arange(self.number_of_particles)
                             + np.random.uniform()) / self.number_of_particles)
        cumsum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < self.number_of_particles:
            if offset_partition[i] < cumsum[j]:
                self.indexes[i] = j
                i += 1
            else:
                j += 1

        self.states[:] = self.states[self.indexes]
        self.weights[:] = self.weights[self.indexes]
        
        if self.pf_method is 'sir':
            '''
             In addition to updating and resampling the position of agents 
             (self.states), we will also resample the speed and gate_out. The
             ideal would be to pass this information on self.states, but this
             would require a change in many parts of the code.
            '''
            #for the hybrid version, the speed and the gate_out are not resampled!!!
            for i in range(self.number_of_particles):
                if (i != self.indexes[i]):
                    model1 = self.models[i]
                    model2 = self.models[self.indexes[i]]
                    for i in range(self.base_model.pop_total):
                        model1.agents[i].speed = model2.agents[i].speed
                        model1.agents[i].loc_desire = model2.agents[i].loc_desire
        
       
        # Could use pool.starmap here, but it's quicker to do it in a single process
        self.models = list(itertools.starmap(ParticleFilter.assign_agents, list(zip(
            range(self.number_of_particles),  # Particle numbers (in integer)
            [s for s in self.states],  # States
            [m for m in self.models]  # Associated Models (a Model object)
        ))))
        return  