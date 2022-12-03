#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:55:42 2022

@author: MichaelKuby
"""

import sys
import string, re
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('reddit averages').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+

def main (in_directory, out_directory):
    text = spark.read.text(in_directory)
    
    wordbreak = r'[%s\s]+' % (re.escape(string.punctuation),) # a regex that matches spaces and/or punctuation
    # takes each row (assumed to be a string) and splits around matches of the given pattern)
    # -1 gives an infinite limit on the number of times the regex can be applied to each string
    words = text.select(functions.split(text.value, wordbreak, -1).alias('words')) # returns a dataframe where column words contains rows of lists of parsed words
    
    # Explode each row of words into their own rows
    exploded = words.select(functions.explode(words.words))
    counts = exploded.groupby('col').count()
    
    # Sort by decreasing count (i.e. frequent words first) and alphabeticcaly if there's a tie
    counts_sorted = counts.orderBy(functions.desc('count'), 'col')
    
    # Remove the count of empy strings
    counts_sorted = counts_sorted.where(counts_sorted['col'] != "")
    
    # Write to CSV
    counts_sorted.write.csv(out_directory)

if __name__ == '__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)