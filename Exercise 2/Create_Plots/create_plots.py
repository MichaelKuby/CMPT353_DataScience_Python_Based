#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 10:35:57 2022

This program plots the distrubtion of views from the first file, and
creates a scatterplot comparing the number of views at hour 12:00 vs
the number of views at hour 13:00 across wikipedia pages. Output generated
is a wikipedia.png file.

@author: MichaelKuby
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename1 = sys.argv[1]
filename2 = sys.argv[2]

# Read from files and create DataFrame's
file1 = pd.read_csv(filename1, sep=' ', header=None, index_col=1, 
            names=['lang', 'page', 'views', 'bytes'])
file2 = pd.read_csv(filename2, sep=' ', header=None, index_col=1, 
            names=['lang', 'page', 'views', 'bytes'])

# Sort filename1 by the number of views (decreasing)
file1 = file1.sort_values(by='views', ascending=False)
print(file1)

"""
Plot 1: Distribution of Views
The following plot shows that the data has a roughly Pareto distribution,
meaning that 20% of the pages accrue 80% of the traffic
"""

plt.figure(figsize=(10, 5)) # changes the size to something sensible
plt.subplot(1, 2, 1) # subplots in 1 row, 2 columns, select the first
plt.plot(file1['views'].values) # builds plot 1

# Use the functions plt.title, plt.xlabel, and plt.ylabel to give some
# useful labels to the plots.
plt.title('Popularity Distribution')
plt.xlabel('Rank')
plt.ylabel('Views')

"""
Plot 2: Hourly Views
A scatterplot of views from the first data file (x-coordinate) and the
corresponding values from the second data file (y-coordinate). We should
expect a linear relationship between those values.
"""

# Sort the values from file2 by views
file2 = file2.sort_values(by='views', ascending=False)
print(file2)

# Get the two series into the same DataFrame.
merged = file1.merge(file2['views'], how='inner', on='page')
print(merged)

# Plot the merged data
plt.subplot(1, 2, 2) # and then select the second
#plt.plot(merged['views_x'].values, merged['views_y'].values) # build plot 2s
plt.scatter(merged['views_x'].values, merged['views_y'].values) # build plot 2s
plt.xscale('log')
plt.yscale('log')

# Use the functions plt.title, plt.xlabel, and plt.ylabel to give some
# useful labels to the plots.
plt.title('Hourly Correlation')
plt.xlabel('Hour 1 views')
plt.ylabel('Hour 2 views')

# Create a file wikipedia.png
plt.savefig('wikipedia.png')