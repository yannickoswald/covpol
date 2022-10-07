# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 16:46:35 2022

@author: earyo
"""

model = CountryModel(0.015, 0.12, 18)
CountryAgent.instances[0]
           ### compute ranges of agent properties (later important for normalization
           ## of distance measure)
max_income = max(agent_data["gdp_pc"]) ##
min_income = min(agent_data["gdp_pc"])
max_politicalregime = max(agent_data["democracy_index"])  ## ~ mean of the distr. as initial seed
min_politicalregime = min(agent_data["democracy_index"])
range_income = max_income - min_income
range_politicalregime = max_politicalregime - min_politicalregime
           #range_covidcases = max_covidcases - min_covidcases
           ## max distance between two points on earth =
           ## earth circumference divided by two
max_distance_on_earth = 40075.017/2
           
           
           ### find all countries that have implemented lockdown already
           ### and store in list

start = dt.now()
list_of_lockdown_countries = [x for x in CountryAgent.instances if x.state == 1]
running_secs = (dt.now() - start).microseconds

      
            #  compute distance function to all other countries that have lockdown
income_differences_array = np.zeros((len(list_of_lockdown_countries)))
politicalregime_differences_array = np.zeros((len(list_of_lockdown_countries)))
#covidcases_differences_array = np.zeros((len(list_of_lockdown_countries), 1))
geo_differences_array = np.zeros((len(list_of_lockdown_countries)))
total_differences_array  = np.zeros((len(list_of_lockdown_countries)))
           
for i in range(0,len(list_of_lockdown_countries)):
                 income_differences_array[i] = CountryAgent.instances[0].income - list_of_lockdown_countries[i].income
                 politicalregime_differences_array[i] = CountryAgent.instances[0].politicalregime - list_of_lockdown_countries[i].politicalregime
                 geo_differences_array[i] = geo_distance(CountryAgent.instances[0].latitude, list_of_lockdown_countries[i].latitude, 
                                                           CountryAgent.instances[0].longitude, list_of_lockdown_countries[i].longitude)
           
total_differences_array  = ( 1/3 * (abs(income_differences_array)) / range_income
                           + 1/3 * (abs(politicalregime_differences_array)) / range_politicalregime
                           + 1/3 * (abs(geo_differences_array)) / max_distance_on_earth)
           
           #total_differences_array = np.delete(total_differences_array, np.where(total_differences_array == 0)[0][0])
           ### the division by max is a normalization on the interval [0,1]
           ### + 0.15 because the random-noise needs to be factored in the maximum range of possible values
           ### 0.15 is roughly calculated here as upper limited (3 std) to np.random.normal 0, 0.05
           ### https://onlinestatbook.com/2/calculators/normal_dist.html
           
total_differences_array = np.sort(total_differences_array)
index_for_choice = 10
CountryAgent.instances[0].minimum_difference = np.mean(total_differences_array[0:index_for_choice])
        


total_differences_array = np.array(
    [  1/3 * abs(CountryAgent.instances[0].income - x.income) / range_income 
     + 1/3 * abs(CountryAgent.instances[0].politicalregime - x.politicalregime) / range_politicalregime
     + 1/3 * (geo_distance(CountryAgent.instances[0].latitude, x.latitude,
          CountryAgent.instances[0].longitude, x.longitude) / max_distance_on_earth)
      for x in CountryAgent.instances if x.state == 1])

  

  

  
income_differences_array = CountryAgent.instances[0].income - [x.income for x in CountryAgent.instances if x.state == 1]
politicalregime_differences_array =  CountryAgent.instances[0].politicalregime - [x.politicalregime for x in CountryAgent.instances if x.state == 1]
geo_differences_array = [geo_distance(CountryAgent.instances[0].latitude, x.latitude, 
                                           CountryAgent.instances[0].longitude, x.longitude) for x in CountryAgent.instances if x.state == 1]

           





model = CountryModel(0.015, 0.12, 18)
CountryAgent.instances[0]
           ### compute ranges of agent properties (later important for normalization
           ## of distance measure)
max_income = max(agent_data["gdp_pc"]) ##
min_income = min(agent_data["gdp_pc"])
max_politicalregime = max(agent_data["democracy_index"])  ## ~ mean of the distr. as initial seed
min_politicalregime = min(agent_data["democracy_index"])
range_income = max_income - min_income
range_politicalregime = max_politicalregime - min_politicalregime
           #range_covidcases = max_covidcases - min_covidcases
           ## max distance between two points on earth =
           ## earth circumference divided by two
max_distance_on_earth = 40075.017/2
           
           
           ### find all countries that have implemented lockdown already
           ### and store in list

start = dt.now()
list_of_lockdown_countries = [x for x in CountryAgent.instances if x.state == 1]
running_secs = (dt.now() - start).microseconds


CountryAgent.instances[0].income - [x.income for x in CountryAgent.instances if x.state == 1]
    
            #  compute distance function to all other countries that have lockdown
            
 
total_differences_array = np.sort(np.array(
    [  1/3 * abs(CountryAgent.instances[0].income - x.income) / range_income 
     + 1/3 * abs(CountryAgent.instances[0].politicalregime - x.politicalregime) / range_politicalregime
     + 1/3 * (geo_distance(CountryAgent.instances[0].latitude, x.latitude,
          CountryAgent.instances[0].longitude, x.longitude) / max_distance_on_earth)
      for x in CountryAgent.instances if x.state == 1]))

  

  

  
income_differences_array = CountryAgent.instances[0].income - [x.income for x in CountryAgent.instances if x.state == 1]
politicalregime_differences_array =  CountryAgent.instances[0].politicalregime - [x.politicalregime for x in CountryAgent.instances if x.state == 1]
geo_differences_array = [geo_distance(CountryAgent.instances[0].latitude, x.latitude, 
                                           CountryAgent.instances[0].longitude, x.longitude) for x in CountryAgent.instances if x.state == 1]

           
total_differences_array  = ( 1/3 * (abs(income_differences_array)) / range_income
                           + 1/3 * (abs(politicalregime_differences_array)) / range_politicalregime
                           + 1/3 * (abs(geo_differences_array)) / max_distance_on_earth)
           
           #total_differences_array = np.delete(total_differences_array, np.where(total_differences_array == 0)[0][0])
           ### the division by max is a normalization on the interval [0,1]
           ### + 0.15 because the random-noise needs to be factored in the maximum range of possible values
           ### 0.15 is roughly calculated here as upper limited (3 std) to np.random.normal 0, 0.05
           ### https://onlinestatbook.com/2/calculators/normal_dist.html
           
total_differences_array = np.sort(total_differences_array)
index_for_choice = 10
CountryAgent.instances[0].minimum_difference = np.mean(total_differences_array[0:index_for_choice])
        
