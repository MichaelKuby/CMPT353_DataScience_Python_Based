#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:05:31 2022

@author: MichaelKuby
"""

import sys
import re
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('wiki popular').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+

wiki_schema = types.StructType([
    types.StructField('language', types.StringType()),
    types.StructField('page', types.StringType()),
    types.StructField('views', types.IntegerType()),
    types.StructField('bytes returned', types.LongType())
])

def pathname_to_hour(pathname):
    # Filenames will be in the format pagecounts-YYYYMMDD-HHMMSS*
    # We want YYYYMMDD-HH substring of that as a label for the day/hour
    find = re.compile('\d{8}-\d{2}')
    substring = re.search(find, pathname).group(0)
    return substring

def main(in_directory, out_directory):
    data = spark.read.csv(in_directory, schema=wiki_schema, sep = " ").withColumn('filename', functions.input_file_name())
    
    # English pages only and remove the main page
    english = data.where(data['page'] != 'Main_Page')
    english = english.where(english['language'] == 'en')
    
    # Remove titles starting with 'Special:'
    english = english.where(functions.substring('page', 1, 8) != 'Special:' )
    
    # Get the date and hour from filenames
    path_to_hour = functions.udf(lambda z:pathname_to_hour(z), returnType=types.StringType()) # creates a User Defined Function to be used on cols
    english = english.withColumn('dates', path_to_hour(english['filename']))
    english.cache()
    
    # Find the largest number of page views in each hour
    largest_views = english.groupBy('dates').agg(functions.max(english['views']))
    
    # Join largest_views with all english results
    results = english.join(largest_views.withColumnRenamed('max(views)', 'views'), ['dates', 'views']) # with help from https://stackoverflow.com/questions/33745964/how-to-join-on-multiple-columns-in-pyspark
    results = results.select('dates', 'page', 'views')

    # Sort by date
    results = results.sort('dates')
    #results.show()
    
    # Output as CSV
    results.write.csv(out_directory, mode='overwrite')
    
if __name__ == '__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)