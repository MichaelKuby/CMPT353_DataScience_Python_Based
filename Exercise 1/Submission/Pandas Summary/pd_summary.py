#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 19:23:15 2022

@author: MichaelKuby
"""

import pandas as pd
totals = pd.read_csv('totals.csv').set_index(keys=['name'])
counts = pd.read_csv('counts.csv').set_index(keys=['name'])

# Find the city with the lowest precipitation over the year
city_totals = totals.sum(axis=1)
print('City with lowest total precipitation: \n', city_totals.idxmin(axis=0))

# Determine the average precipitation in these locations for each month. 
# That will be the total precipitation for each month (axis 0), divided 
# by the total observations for that months. Print the resulting array.
monthly_prec = totals.sum(axis = 0)
monthly_obs = counts.sum(axis=0)
average_prec_by_loc = monthly_prec / monthly_obs
print ('Average precipitation in each month: \n', average_prec_by_loc)

# Do the same for the cities: give the average precipitation (daily precipitation
# averaged over the month) for each city by printing the array.
city_obs = counts.sum(axis=1)
city_avg_prec = city_totals / city_obs
print ('Average precipitation in each city: \n', city_avg_prec)