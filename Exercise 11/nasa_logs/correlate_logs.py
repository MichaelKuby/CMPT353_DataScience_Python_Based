import sys
import math
assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
from pyspark.sql import SparkSession, functions, types, Row
import re


line_re = re.compile(r"^(\S+) - - \[\S+ [+-]\d+\] \"[A-Z]+ \S+ HTTP/\d\.\d\" \d+ (\d+)$")


def line_to_row(line):
    """
    Take a logfile line and return a Row object with hostname and bytes transferred.
    Return None if regex doesn't match.
    """
    m = line_re.match(line)
    if m:
        return Row(hostname=m.group(1), bytes=m.group(2))
    else:
        return None


def not_none(row):
    """
    Is this None? Hint: .filter() with it.
    """
    return row is not None


def create_row_rdd(in_directory):
    log_lines = spark.sparkContext.textFile(in_directory)
    # TODO: return an RDD of Row() objects
    return log_lines.map(line_to_row).filter(not_none)

def main(in_directory):
    logs = spark.createDataFrame(create_row_rdd(in_directory))
    totals = logs.groupBy('hostname') \
        .agg(functions.count('hostname').alias('count_requests'), \
             functions.sum('bytes').alias('sum_request_bytes'))
    totals = totals.withColumn('count_requests^2', totals['count_requests'] * totals['count_requests'])
    totals = totals.withColumn('sum_request_bytes^2', totals['sum_request_bytes'] * totals['sum_request_bytes'])
    totals = totals.withColumn('count_requests * sum_request_bytes', totals['count_requests'] * totals['sum_request_bytes'])
    
    # TODO: calculate r.
    values = totals.groupBy().agg(functions.count('hostname').alias('num_rows'),
                                  functions.sum('count_requests').alias('xi'),
                                  functions.sum('sum_request_bytes').alias('yi'),
                                  functions.sum('count_requests^2').alias('xi^2'),
                                  functions.sum('sum_request_bytes^2').alias('yi^2'),
                                  functions.sum('count_requests * sum_request_bytes').alias('xi*yi'))
    
    values = values.first()
    
    n = values[0]
    sum_x = values[1]
    sum_y = values[2]
    sum_x_sqrd = values[3]
    sum_y_sqrd = values[4]
    sum_x_times_y = values[5]
    
    # let r = (
    
    num = (n * sum_x_times_y) - (sum_x * sum_y)
    denom = math.sqrt((n * sum_x_sqrd) - (sum_x * sum_x)) * math.sqrt((n * sum_y_sqrd) - (sum_y * sum_y))
    
    r = num / denom
    
    print(f"r = {r}\nr^2 = {r*r}")
    
    # Built-in function should get the same results.
    # print(totals.corr('count_requests', 'sum_request_bytes'))


if __name__=='__main__':
    in_directory = sys.argv[1]
    spark = SparkSession.builder.appName('correlate logs').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.sparkContext.setLogLevel('WARN')

    main(in_directory)
