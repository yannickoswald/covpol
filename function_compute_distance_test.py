# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 09:53:02 2022

@author: earyo
"""

 
           ### find all countries that have implemented lockdown already
           ### and store in list
           ### also compute ranges of agent properties
            
max_income = 20000 ## ~ mean of the distr. as initial seed
min_income = 0 
           
max_politicalregime = 5.43  ## ~ mean of the distr. as initial seed
min_politicalregime = 0 
           
max_covidcases = 0.4 ## ~ mean of the distr. as initial seed
min_covidcases = 0
           

list_of_lockdown_countries = []
for i in range(0, len(CountryAgent.instances)):     
               if CountryAgent.instances[i].state == 1:
                  list_of_lockdown_countries.append(CountryAgent.instances[i])
               
                
         ## find min and max income
               if CountryAgent.instances[i].income > max_income:
                       max_income = CountryAgent.instances[i].income
               if CountryAgent.instances[i].income < min_income:
                       min_income = CountryAgent.instances[i].income
                       
               ## find min and max politicalregime        
               if CountryAgent.instances[i].politicalregime > max_politicalregime:
                      max_politicalregime = CountryAgent.instances[i].politicalregime
               if CountryAgent.instances[i].income < min_politicalregime:
                      min_politicalregime = CountryAgent.instances[i].politicalregime
                        
               ## find min and max covidcases       
               if CountryAgent.instances[i].covidcases > max_covidcases:
                      max_covidcases = CountryAgent.instances[i].covidcases
               if CountryAgent.instances[i].covidcases < min_covidcases:
                      min_covidcases = CountryAgent.instances[i].covidcases
                       
range_income = max_income - min_income
range_politicalregime = max_politicalregime - min_politicalregime
range_covidcases = max_covidcases - min_covidcases
               
           #  compute distance function to all other countries that have lockdown
           
           
income_differences_array = np.zeros((len(list_of_lockdown_countries), 1))
politicalregime_differences_array = np.zeros((len(list_of_lockdown_countries), 1))
covidcases_differences_array = np.zeros((len(list_of_lockdown_countries), 1))
geo_differences_array = np.zeros((len(list_of_lockdown_countries), 1))
total_differences_array  = np.zeros((len(list_of_lockdown_countries), 1))
           

           

for i in range(0,len(list_of_lockdown_countries)):
                income_differences_array[i,0] = CountryAgent.instances[11].income - list_of_lockdown_countries[i].income
                politicalregime_differences_array[i,0] = CountryAgent.instances[11].politicalregime - list_of_lockdown_countries[i].politicalregime
                covidcases_differences_array[i,0] = CountryAgent.instances[11].covidcases - list_of_lockdown_countries[i].covidcases
                geo_differences_array[i,0] = np.sqrt((CountryAgent.instances[11].pos[0] - list_of_lockdown_countries[i].pos[0]) ** 2 + (CountryAgent.instances[11].pos[1] - list_of_lockdown_countries[i].pos[1]) ** 2)

total_differences_array  = ((abs(income_differences_array)) / range_income
                                      + (abs(politicalregime_differences_array)) / range_politicalregime
                                      + (abs(covidcases_differences_array)) / range_covidcases
                                      + (abs(geo_differences_array)) / np.sqrt( 20 * 20 + 10 * 10 ))
                                      
           #total_differences_array = np.delete(total_differences_array, np.where(total_differences_array == 0)[0][0])
           
           
CountryAgent.instances[11].minimum_difference = ((min(total_differences_array, default=0)
                                                + np.random.normal(0,0.05)) 
                                                / (max(total_differences_array, default = 0) + 0.15))
           ### the division by max is a normalization on the interval [0,1]
           ### + 0.15 because the random-noise needs to be factored in the maximum range of possible values
           ### 0.15 is roughly calculated here as upper limited (3 std) to np.random.normal 0, 0.05 
           ### https://onlinestatbook.com/2/calculators/normal_dist.html
