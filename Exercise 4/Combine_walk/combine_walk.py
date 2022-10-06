#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 13:12:56 2022

@author: MichaelKuby
"""

import os
import pathlib
import sys
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation, parse
    xmlns = 'http://www.topografix.com/GPX/1/0'
    
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.10f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.10f' % (pt['lon']))
        time = doc.createElement('time')
        time.appendChild(doc.createTextNode(pt['datetime'].strftime("%Y-%m-%dT%H:%M:%SZ")))
        trkpt.appendChild(time)
        trkseg.appendChild(trkpt)

    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)

    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)

    doc.documentElement.setAttribute('xmlns', xmlns)

    with open(output_filename, 'w') as fh:
        fh.write(doc.toprettyxml(indent='  '))


def get_data(input_gpx):
    """
    
    Parameters
    ----------
    input_gpx : .gpx file
        .gpx file from a goPro

    Returns
    -------
    points : pandas DataFrame
        Columns: lat, lon, datetime

    """
    tree = ET.parse(input_gpx)
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
    
    return points


def main():
    # Access directories
    input_directory = pathlib.Path(sys.argv[1])
    output_directory = pathlib.Path(sys.argv[2])
    
    # accl and gps are goPro data
    accl = pd.read_json(input_directory / 'accl.ndjson.gz', lines=True, convert_dates=['timestamp'])[['timestamp', 'x']]
    gps = get_data(input_directory / 'gopro.gpx')
    # phone is phone data
    phone = pd.read_csv(input_directory / 'phone.csv.gz')[['time', 'gFx', 'Bx', 'By']]

    # Assume the phone data starts at the same time as the accelerometer data
    # Create a timestamp in phone based off of this assumption.
    first_time = accl['timestamp'].min()
    phone['timestamp'] = first_time + pd.to_timedelta(phone['time'], unit = 'sec')
    
    # Aggregate timestamps into 4-second bins
    phone['timestamp'] = pd.Series(phone['timestamp']).dt.round('4S')
    accl['timestamp'] = pd.Series(accl['timestamp']).dt.round('4S')
    gps.index = pd.Series(gps.index).dt.round('4s')
    
    # Group on the rounded times and average all other columns
    phone = phone.groupby('timestamp').mean()
    accl = accl.groupby('timestamp').mean()
    gps = gps.groupby(gps.index).mean()
    
    # Join the three DataFrames on the timestamp index
    combined = pd.merge(phone, accl, how='outer', left_index= True, right_index=True)
    combined = pd.merge(combined, gps, how='outer', left_index= True, right_index=True)
    combined = combined.dropna()
    
    """ 
    The timestamps in accl and gps will be accurate, since they are both
    from the GoPro; however, the timestamp in phone will be slightly
    inaccurate since record was not pressed at the same time - despite
    the fact that earlier we assumed that they were.
    
    Luckily, the same fields were measured in both devices. We want to 
    look at acceleration in the x-axis from the phone data (gFx) and
    acceleration from the accelerometer (x). We want to find a time offset
    for the phone data (gFx) that leads to an alignment of the two columns
    sets of values.
    
    We're told the offset will be at most 5 seconds, and must be accurate
    to one decimal place. I.e., offset will be a value chosen from
    np.linspace(-5.0, 5.0, 101)
    
    Note: the goal here is to get the most accurate lat and lon from gps 
    """
    # Get the original phone data and put it in a new dataframe
    phone2 = pd.read_csv(input_directory / 'phone.csv.gz')[['time', 'gFx', 'Bx', 'By']] 
    
    best = float() # for comparison
    accl = accl.reset_index()
    
    # Apply the offset, do the 4 second rounding/grouping/averaging and cross-correlation
    for offset in np.linspace(-5.0, 5.0, 101):
        # Create a timestamp in phone based off of the offset
        phone2['timestamp'] = first_time + pd.to_timedelta(phone2['time'].copy() + offset, unit = 'sec')
        phone3 = phone2.copy()
        # Aggregate timestamps into 4-second bins
        phone3['timestamp'] = pd.Series(phone3['timestamp']).dt.round('4S')
        phone3 = phone3.groupby('timestamp').mean()
        phone3 = phone3.reset_index()
        #Compute cross correlation on the current offset with the accl data
        val = np.correlate(phone3['gFx'], accl['x'])
        if val.max() > best:
            best = val.max()
            best_offset = offset
    
    phone2['timestamp'] = first_time + pd.to_timedelta(phone2['time'].copy() + best_offset, unit = 'sec') # synchronized
    phone2['timestamp'] = pd.Series(phone2['timestamp']).dt.round('4S')
    phone2 = phone2.groupby('timestamp').mean()
    
    # Set index for accl back to timestamp
    accl = accl.set_index('timestamp')
    
    # Join the three DataFrames on the timestamp index
    combined2 = pd.merge(phone2, accl, how='right', left_index= True, right_index=True)
    combined2 = pd.merge(combined2, gps, how='outer', left_index= True, right_index=True)
    combined2 = combined2.dropna()
    combined2 = combined2.reset_index()
    combined2 = combined2.rename(columns={'timestamp': 'datetime'})
    
    print(f'Best time offset: {best_offset:.1f}')
    
    os.makedirs(output_directory, exist_ok=True)
    output_gpx(combined2[['datetime', 'lat', 'lon']], output_directory / 'walk.gpx')
    combined2[['datetime', 'Bx', 'By']].to_csv(output_directory / 'walk.csv', index=False)
    
main()