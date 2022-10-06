#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 10:20:15 2022

@author: MichaelKuby
"""

import sys
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pykalman import KalmanFilter
from math import pi

def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.7f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.7f' % (pt['lon']))
        trkseg.appendChild(trkpt)
    
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')

def distance(data):
    # Calculate Distances between points using the Haversine calculation
    """
    The following code influenced by https://www.youtube.com/watch?v=HaGj0DjX8W8 
    and by Greg Baker: https://coursys.sfu.ca/2022fa-cmpt-353-d1/forum/77
    """
    data['lat1'] = data['lat'].shift(-1)
    data['lon1'] = data['lon'].shift(-1)
    data = data.dropna().copy()
    R = 3440.1 # Earth's radius in NM (nautical miles)
    p = pi/180
    C = 1852 # Where 1852 converts Nautical Miles to meters
    data['distance'] = ( 
        R * np.arccos( (np.sin(data['lat']*p) * np.sin(data['lat1']*p) + 
                        np.cos(data['lat']*p) * np.cos(data['lat1']*p) * 
                        np.cos(data['lon']*p - data['lon1']*p)))) * C 
    result = np.sum(data['distance'])
    return result

def main():
    # Get the args from the command line
    file1 = sys.argv[1] # GPX
    file2 = sys.argv[2] # CSV
    
    # Read the XML file using xml element tree
    tree = ET.parse(file1)
    
    pos_rows = []
    time_rows = []
    
    # Get the latitude and longitude from each trkpt node
    for node in tree.iter('{http://www.topografix.com/GPX/1/0}trkpt'):
        s_lat = float(node.attrib.get('lat'))
        s_lon = float(node.attrib.get('lon'))
        pos_rows.append({'lat': s_lat, 'lon': s_lon})
    
    # Put those rows into a dataframe
    """
    From XML to dataframes done with insight from 
    https://medium.com/@robertopreste/from-xml-to-pandas-dataframes-9292980b1c1c
    """
    df1_cols = ['lat', 'lon']    
    df1 = pd.DataFrame(pos_rows, columns = df1_cols)
    
    # Get the time in text from each time node    
    for time in tree.iter('{http://www.topografix.com/GPX/1/0}time'):
        s_time = time.text
        time_rows.append({'datetime': s_time})
    
    # Put those rows into a dataframe
    df2_cols = ['datetime']
    df2 = pd.DataFrame(time_rows, columns = df2_cols)
    
    # Merge all of our data into one dataframe
    points = pd.merge(df1, df2, how = 'inner', left_index=True, right_index=True)
    
    # Convert date Series to datetime objects
    points['datetime'] = pd.to_datetime(points['datetime'], utc=True)
    points = points.set_index('datetime')
    
    # Read the CSV using pandas
    sensor_data = pd.read_csv(file2, parse_dates=['datetime']).set_index('datetime')
    points['Bx'] = sensor_data['Bx']
    points['By'] = sensor_data['By']
    
    
    dist = distance(points.copy())
    output1 = f'Unfiltered distance: {dist:.2f}'
    print(output1)
    
    """
    Kalman filtering
    What are my variables (the variables make up the state)? 
    latitude, longitude, SensorBx, and SensorBy
    
    Each of these variables has a mean (most likely 'truth' value) 
    and a variance (uncertainty)
    """
    
    # What is my best guess for the 'true' intitial state?
    initial_state = points.iloc[0]
    
    # Observation covariance: what are the standard deviations for each of the variables that make up my state?
    observation_covariance = np.diag([.02, .02, 2, 2]) ** 2
    
    # Transition covariance: what are the predicted standard deviations for each of the variables that make up my prediction matrix?
    # These values were chosen based on the idea that our predicted GPS coordinates are likely to have a similar
    # or slightly smaller error as that compared to the observed values, but that our sensor estimates are similar or worse
    # since we are predicting that they don't change.
    transition_covariance = np.diag([.01, 0.01, 5, 5]) ** 2
    
    """
    What are my predictions for the next value based on the current values?
    
    latitude = latitude + (6 x 10^-7 x Bx) + (29 x 10^-7 x By)
    longitude = longitude + (-43 x 10^-7 x Bx) + (12 x 10 ^ -7 x By)
    Bx = Bx
    By = By
    """
    transition = [[1,0, 6e-7, 29e-7], [0,1, -43e-7, 12e-7], [0,0,1,0], [0,0,0,1]]
    
    kf = KalmanFilter(
        initial_state_mean = initial_state,
        initial_state_covariance = observation_covariance,
        observation_covariance = observation_covariance,
        transition_covariance = transition_covariance,
        transition_matrices = transition
    )
    
    smoothed_points, _ = kf.smooth(points)
    smoothed_points = pd.DataFrame(smoothed_points)
    smoothed_points = smoothed_points.rename(columns={0: 'lat', 1: 'lon', 2: 'Bx', 3: 'By'})
    dist = distance(smoothed_points)
    output2 = f'Filtered distance: {dist:.2f}'
    print(output2)
    
    with open('calc_distance.txt', 'w') as f:
        f.write(output1)
        f.write('\n')
        f.write(output2)
        f.close()
        
    output_gpx(smoothed_points, 'out.gpx')  # View this file against an unsmoothed file via gpx.studio

if __name__ == '__main__':
    main()
