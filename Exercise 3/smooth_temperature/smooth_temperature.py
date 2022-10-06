#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 13:11:12 2022

@author: MichaelKuby
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter

# Get the data
filename = sys.argv[1]
cpu_data = pd.read_csv(filename, header=0)

# Convert date Series to datetime objects
cpu_data['timestamp'] = pd.to_datetime(cpu_data['timestamp'])

# Apply LOESS curve smoothening technique.
filtered = lowess(cpu_data['temperature'], cpu_data['timestamp'], frac=.02)

"""
Kalman filtering
What are my variables (the variables make up the state)? 
Temperature, cpu_percent, system load, and rpm

Each of these variables has a mean (most likely 'truth' value) 
and a variance (uncertainty)
"""

kalman_data = cpu_data[['temperature', 'cpu_percent', 'sys_load_1', 'fan_rpm']]

# What is my best guess for the 'true' intitial state?
initial_state = kalman_data.iloc[0]

# Observation covariance: what are the standard deviations for each of the variables that make up my state?
# These values were obtained by using cpu_data['column'].describe()
observation_covariance = np.diag([0.1, 0.05, 0.46, 15]) ** 2

# Transition covariance: what are the predicted standard deviations for each of the variables that make up my prediction matrix?
# These values were chosen based on the idea that our prediction is likely to have a similar
# or slightly smaller error as that compared to the observed values.
transition_covariance = np.diag([.07, 0.05, 0.46, 15]) ** 2

"""
What are my predictions for the next value based on the current values?

temperature = 0.96 * temperature + 0.5 x cpu_percent + 0.2 * sys_load_1 - 0.001 x fan_rpm
cpu_percent = 0.1 * temperature + 0.4 x cpu_percent _ 2.3 * sys_load_1
sys_load_1 = 0.96 * sys_load_1
fan_rpm = fan_rpm
"""
transition = [[0.96, 0.5, 0.2, -0.001], [0.1 ,0.4 , 2.3, 0], [0,0, 0.96,0], [0,0,0,1]]

kf = KalmanFilter(
    initial_state_mean = initial_state,
    initial_state_covariance = observation_covariance,
    observation_covariance = observation_covariance,
    transition_covariance = transition_covariance,
    transition_matrices = transition
)

pred_state, state_cov = kf.smooth(kalman_data)


# Plot the results
plt.figure(figsize=(12, 4))
plt.plot(cpu_data['timestamp'], cpu_data['temperature'], 'b.', alpha=0.5);
plt.plot(cpu_data['timestamp'], filtered[:, 1], 'r-', linewidth=3)
plt.plot(cpu_data['timestamp'], pred_state[:, 0], 'g-')
plt.legend(['Data Points', 'LOESS Smoothening', 'Kalman Filter'])
plt.ylabel('Temperature')
plt.xlabel('Time (Month-Day Hour)')
plt.savefig('cpu.svg')