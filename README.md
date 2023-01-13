# Covid_policy_response_abm

This repository is for the publication 

Oswald, Malleson, Suchak (2023). An agent-based model of the 2020 international policy 
diffusion in response to the COVID-19 pandemic with particle filter. (in preparation)

To reproduce the full body of work, take the following steps:

1. Run the jupyter "notebook run_base_model_and_filter_with_plotting_jupyter.ipynb"
   This reproduces a substantial amount of the above paper including Figure 2, 4 and 5.
   To reproduce Figure 4 exactly, the notebook has to take the parameter no_of_iterations = 100.
   To reproduce Figure 5 exactly, the notebeook has to take the parameter n_of_iterations = 1000.
   
2. To reproduce Figure 6 in full several intermediate steps are necessary: 
    
    2.1 Run the script file "number_of_particles_experiment_MSE.py" to reproduce the data points where iterations = 20.
    
    2.2 To reproduce the datapoints where iterations = 1, run the following:
    
         2.2.1 "particle_filter_only.py" and collect the data.
         
         2.2.2 "run_base_model_only_parallelized" and collect the data.
    
    Best run from anaconda command prompt.
    
3. To reproduce Figure 7 run the script "experiment_da_window_size.py". Best run from anaconda command prompt.
