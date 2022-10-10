import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


OUTPUT_TEMPLATE = (
    "Initial T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann-Whitney U-test p-value: {utest_p:.3g}"
)


def main():
    reddit_counts = sys.argv[1]
    counts = pd.read_json(reddit_counts, lines = True)
    
    # filter by dates including years 2012 and 2013 only, and by the subreddit Canada
    df_2012_13 = counts[(counts['date'].dt.year >= 2012) & (counts['date'].dt.year < 2014) & (counts['subreddit'] == 'canada')]
    df_2012_13_wkday = df_2012_13[df_2012_13['date'].dt.dayofweek < 5]
    df_2012_13_wkend = df_2012_13[df_2012_13['date'].dt.dayofweek >4]
    
    # Use scipy.stats to do a T-test on the data to get a p-value.
    # Can we conclude that there are a different number of comments on weekdays
    # compared to weekends?
    
    plt.hist(df_2012_13_wkday['comment_count'])
    plt.hist(df_2012_13_wkend['comment_count'])
    plt.show()
    ttest = stats.ttest_ind(df_2012_13_wkday['comment_count'], df_2012_13_wkend['comment_count'])
    print(ttest)

    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p=0,
        initial_weekday_normality_p=0,
        initial_weekend_normality_p=0,
        initial_levene_p=0,
        transformed_weekday_normality_p=0,
        transformed_weekend_normality_p=0,
        transformed_levene_p=0,
        weekly_weekday_normality_p=0,
        weekly_weekend_normality_p=0,
        weekly_levene_p=0,
        weekly_ttest_p=0,
        utest_p=0,
    ))


if __name__ == '__main__':
    main()
