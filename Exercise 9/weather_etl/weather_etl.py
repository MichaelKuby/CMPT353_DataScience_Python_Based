import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('weather ETL').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+

observation_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.StringType()),
    types.StructField('observation', types.StringType()),
    types.StructField('value', types.IntegerType()),
    types.StructField('mflag', types.StringType()),
    types.StructField('qflag', types.StringType()),
    types.StructField('sflag', types.StringType()),
    types.StructField('obstime', types.StringType()),
])


def main(in_directory, out_directory):

    weather = spark.read.csv(in_directory, schema=observation_schema)
    
    # Keep only where qflag is null, station starts with 'CA', and observation is 'TMAX'
    data = weather.filter(weather.qflag.isNull())
    data = data.filter(data['station'].startswith('CA'))   
    data = data.filter(data['observation'] == 'TMAX')
    
    # Divide the temparature by 10 so it's actually in Celcius, and call the column tmax
    data = data.withColumn('tmax', data['value'] / 10)
    
    #Keep only the columns station, date, and tmax
    cleaned_data = data.select(
        data['station'],
        data['date'],
        data['tmax'])
    
    cleaned_data.write.json(out_directory, compression='gzip', mode='overwrite')

if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
