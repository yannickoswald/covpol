# COVPOL: An agent-based model of the international COVID-19 policy response

This repository is for the publication 

Oswald, Malleson, Suchak (2023). An agent-based model of the 2020 international policy 
diffusion in response to the COVID-19 pandemic with particle filter 

available as *preprint* at https://arxiv.org/abs/2302.11277

This work has been implemented using the Python Anaconda distribution and the agent-based model package MESA in particular.
It is provided with a `.yml` file specifying a conda environment which contains the required packages.
In order to set up the environment, run the following command from the
terminal/conda prompt:
```
conda env create -f env.yml
```
This will create a new conda environment titled `cov-pol`.
The environment can then be activated using the following command:
```
conda activate cov-pol
```

To reproduce the full body of work, take the following steps:

1. Run the script file [run_base_model_and_filter_with_plotting.py](./covpol/run_base_model_and_filter_with_plotting.py) in the `covpol` directory as follows:
   ```
   cd covpol
   python run_base_model_and_filter_with_plotting.py
   ```
   This reproduces a substantial amount of the above paper including Figure 2, 4 and 5.
   To reproduce Figure 4 exactly, the notebook has to take the parameter `no_of_iterations = 100`.
   To reproduce Figure 5 exactly, the notebeook has to take the parameter `no_of_iterations = 1000`.
   
3. To reproduce Figure 6 in full several intermediate steps are necessary (time-expensive): 

    1. Run the script file `number_of_particles_experiment_MSE.py` to reproduce the data points where iterations = 20.
    2. To reproduce the datapoints where iterations = 1, run the following:
    
        1. `particle_filter_only.py` and collect the data.
        2. `run_base_model_only_parallelized.py` and collect the data.
    
    3. Alternatively Figure 6 can be reproduced exactly in a time-cheap manner as it is from the script file `graph_mse_number_of_particles_experiment.py`.
     
     Best run from anaconda command prompt.
    
4. To reproduce Figure 7 run the script `experiment_da_window_size.py` (time-expensive). Best run from anaconda command prompt.
