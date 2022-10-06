#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 16:20:48 2022

@author: MichaelKuby
"""

import sys
import pandas as pd
import numpy as np
from math import pi
import matplotlib.pyplot as plt

def distance(city, stations):
    # city should be a row of data accessed via index
    # stations is the entire dataframe
    R = 3440.1 # Earth's radius in NM (nautical miles)
    p = pi/180
    C = 1852 # Where 1852 converts Nautical Miles to meters
    distances = ( 
        R * np.arccos( (np.sin(city['latitude']*p) * np.sin(stations['latitude']*p) + 
                        np.cos(city['latitude']*p) * np.cos(stations['latitude']*p) * 
                        np.cos(city['longitude']*p - stations['longitude']*p)))) * C 
    return distances.idxmin()

def best_tmax(city, stations):
    tmax = stations.loc[city]['avg_tmax']
    return tmax

def main():
    # Import the data
    stations = pd.read_json(sys.argv[1], lines=True)
    city_data = pd.read_csv(sys.argv[2])
    city_data = city_data.dropna()
    
    """
    stations['avg_tmax'] is Celsius X 10, since that's what the Global
    Historical Climatology Network (GHCN) distributes. Hence we must divide by 10.
    """
    stations['avg_tmax'] = stations['avg_tmax'] / 10 # Now in Celsius
    
    """
    1. Area is given in meters squared; convert to km squared
    2. Exclude cities with area greater than 10,000 km squared (unreasonable)
    Note: Population density is population divided by area.
    """
    
    # 1. 1 square meter = 1e6 square kilometer
    city_data['area'] = city_data['area'] / 1e6
    
    # 2. Exclude cities with area greater than 10,000 km squared
    # With help from:
    # https://www.geeksforgeeks.org/how-to-drop-rows-in-dataframe-by-conditions-on-column-values/
    drop_indices = city_data[ (city_data['area'] > 10000) ].index
    city_data = city_data.drop(drop_indices)
    
    # Population density is pop / area
    city_data['pop_density'] = city_data['population'] / city_data['area']
    
    """
    Find the weather station that is closest to each city and pair it's
    avg_tmax value with that city.

    Will be an O(mn) kind of calc: the distance between every city and 
    station pair must be calculated.

    Steps:
        
        1. Write a function distance(city, stations) that calculates the
        distance between one city and every station.
        
        2. Write a function best_tmax(city, stations) that returns the best
        value you can find for avg_tmax for that one city, from the list
        of all stations
        
        3. Apply across all cities. Hint(cities.apply(best_tmax, stations=stations))

    Make sure to use python operators on Series/Arrays or NumPy ufuncs.

    If same minimum distance to multiple stations, use the station that is first
    in the input data. This matches the behaviour of .argmin
    """
    # Step 1 uses func distance
    closest = city_data.apply(distance, axis = 1, stations=stations)
    # Step 2 uses func best_tmax
    temps = closest.apply(best_tmax, stations=stations)
    
    # Move the data into the appropriate dataframe
    city_data['avg_temp'] = temps
    
    # Graph using a scatterplot and output via command line arg
    plt.scatter(city_data['avg_temp'], city_data['pop_density'], linewidths=0.5, edgecolors='none')
    plt.xlabel('Avg Max Temperature (\u00b0C)')
    plt.ylabel('Population Density (people/km\u00b2)')
    plt.title('Temperature vs Pppulation Density')
    output_file = sys.argv[3]
    plt.savefig(output_file)
    
main()
