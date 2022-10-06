#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 21:09:25 2022

@author: MichaelKuby
"""

# Import numpy and get the data from the numpy file
import numpy as np
data = np.load('monthdata.npz')
totals = data['totals']
observations = data['counts']

# Find the city with the lowest precipitation over the year
city_totals = totals.sum(axis=1) # sum across axis 1
min_prec_city = city_totals.argmin() # use argmin to find the min value in the array
print ('Row with lowest total precipitation: \n', min_prec_city)

# Determine the average precipitation in these locations for each month. 
# That will be the total precipitation for each month (axis 0), divided 
# by the total observations for that months. Print the resulting array.
monthly_prec = totals.sum(axis = 0)
monthly_obs = observations.sum(axis=0)
average_prec_by_loc = monthly_prec / monthly_obs
print ('Average precipitation in each month: \n', average_prec_by_loc)

# Do the same for the cities: give the average precipitation (daily precipitation
# averaged over the month) for each city by printing the array.

city_obs = observations.sum(axis=1)
city_avg_prec = city_totals / city_obs
print ('Average precipitation in each city: \n', city_avg_prec)

# Calculate the total precipitation for each quarter in each city (i.e. the 
# totals for each station across three-month groups). You can assume the number 
# of columns will be divisible by 3. Hint: use the reshape function to reshape 
# to a 4n by 3 array, sum, and reshape back to n by 4.
totals2 = totals.reshape(36, 3)
totals2_sum = totals2.sum(axis=1)
quarterly_prec = totals2_sum.reshape(9, 4)
print('Quarterly precipitiation totals: \n', quarterly_prec)