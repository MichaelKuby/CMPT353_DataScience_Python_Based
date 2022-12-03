import sys
assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
from pyspark.sql import SparkSession, functions, types

comments_schema = types.StructType([
    types.StructField('archived', types.BooleanType()),
    types.StructField('author', types.StringType()),
    types.StructField('author_flair_css_class', types.StringType()),
    types.StructField('author_flair_text', types.StringType()),
    types.StructField('body', types.StringType()),
    types.StructField('controversiality', types.LongType()),
    types.StructField('created_utc', types.StringType()),
    types.StructField('distinguished', types.StringType()),
    types.StructField('downs', types.LongType()),
    types.StructField('edited', types.StringType()),
    types.StructField('gilded', types.LongType()),
    types.StructField('id', types.StringType()),
    types.StructField('link_id', types.StringType()),
    types.StructField('name', types.StringType()),
    types.StructField('parent_id', types.StringType()),
    types.StructField('retrieved_on', types.LongType()),
    types.StructField('score', types.LongType()),
    types.StructField('score_hidden', types.BooleanType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('subreddit_id', types.StringType()),
    types.StructField('ups', types.LongType()),
    #types.StructField('year', types.IntegerType()),
    #types.StructField('month', types.IntegerType()),
])


def main(in_directory, out_directory):
    comments = spark.read.json(in_directory, schema=comments_schema).cache() # since we need this to create averages as well as joined
    
    # TODO: calculate averages
    averages = comments.groupBy('subreddit').agg(functions.mean('score')).cache()
    
    # Exclude subreddits with average score <= 0
    averages_pos = averages.select(
        averages['subreddit'],
        averages['avg(score)']).where(averages['avg(score)'] > 0)
    
    # Join the average score to the collection of all comments and divide to get the score relative to the average.
    joined = comments.join(averages_pos.hint('broadcast'), on=['subreddit'])
    joined = joined.withColumn('relative_score', joined['score']/joined['avg(score)']) 
    joined.cache() # since joined is used to create and then join with max_rel_scores
    
    # Determine the max relative score for each subreddit
    max_rel_scores = joined.groupBy('subreddit').agg(functions.max('relative_score'))
    
    
    # Join again to get the best comment on each subreddit to get the author
    joined = joined.join(max_rel_scores.hint('broadcast'), on=['subreddit'])
    best_author = joined.select(
        'subreddit',
        'author',
        'relative_score').where(joined['relative_score'] == joined['max(relative_score)']).withColumnRenamed('relative_score', 'rel_score')
    
    # write output to directory
    best_author.write.json(out_directory, mode='overwrite')
    
    """
    With broadcast: 48 seconds
    Without broadcast: 2 min 28 seconds
    """
    
if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    spark = SparkSession.builder.appName('Reddit Relative Scores').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.sparkContext.setLogLevel('WARN')

    main(in_directory, out_directory)
