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
    wkday_counts = df_2012_13[df_2012_13['date'].dt.dayofweek < 5].copy()
    wkend_counts = df_2012_13[df_2012_13['date'].dt.dayofweek >4].copy()
    
    # Use scipy.stats to do a T-test on the data to get a p-value.
    # Can we conclude that there are a different number of comments on weekdays
    # compared to weekends?
    
    # Null hypothesis is that "there is not a difference in the number of comments on weekdays vs weekends
    ttest = stats.ttest_ind(wkday_counts['comment_count'], wkend_counts['comment_count'])
    print(ttest) # based on the p-value alone, we would conclude that there is a difference (p-value is very small!)
    print()
    
    # Null hypothesis for a normal test is that "the data is normally distributed"
    # We actually WANT p > 0.05 so that we do not have to reject the null.
    ogwkday_normality_pval = stats.normaltest(wkday_counts['comment_count']).pvalue
    ogwkend_normality_pval = stats.normaltest(wkend_counts['comment_count']).pvalue
    print("Normal test p value for weekdays comment counts:", ogwkday_normality_pval) # p is SMALL so we have to reject the null! The data is not normally distributed.
    print("Normal test p value for weekends comment counts:", ogwkend_normality_pval) # p is SMALL so we have to reject the null! The data is not normally distributed.
    
    # Testing for equal variances. Null hypothesis: two two data sets have equal variances
    # If the p value is small we reject the null hypothesis and conclude that the two data sets have different variances.
    initial_levene = stats.levene(wkday_counts['comment_count'], wkend_counts['comment_count']).pvalue
    print ("Levene (variance) test p value for weekdays and weekends:", initial_levene) # p value is .0438 so we must reject
    print()
    
    plt.hist(wkday_counts['comment_count'])
    plt.title("Weekdays")
    plt.show()
    plt.hist(wkend_counts['comment_count'])
    plt.title("Weekends")
    plt.show()
    
    # Transform the data to try to normalize it. Options: np.sqrt, np.log, np.exp, or squared (**2)
    # Try sqrt
    
    wkday_counts['comment_count_sqrt'] = np.sqrt(wkday_counts['comment_count'])
    wkend_counts['comment_count_sqrt'] = np.sqrt(wkend_counts['comment_count'])
    
    transformed_weekday_normality_p = stats.normaltest(wkday_counts['comment_count_sqrt']).pvalue
    transformed_weekend_normality_p = stats.normaltest(wkend_counts['comment_count_sqrt']).pvalue
    transformed_levene_p = stats.levene(wkday_counts['comment_count_sqrt'], wkend_counts['comment_count_sqrt']).pvalue
    print("Normal test p value for weekdays comment counts square root:", transformed_weekday_normality_p) # Cannot reject
    print("Normal test p value for weekends comment counts square root:", transformed_weekend_normality_p) # Can reject
    print ("Levene (variance) test p value for weekdays and weekends square root:", transformed_levene_p)
    print()
    
    # Try log
    
    wkday_counts['comment_count_log'] = np.log(wkday_counts['comment_count']) 
    wkend_counts['comment_count_log'] = np.log(wkend_counts['comment_count'])
    
    print("Normal test p value for weekdays comment counts log:", stats.normaltest(wkday_counts['comment_count_log']).pvalue) # Cannot reject
    print("Normal test p value for weekends comment counts log:", stats.normaltest(wkend_counts['comment_count_log']).pvalue) # Can reject
    print ("Levene (variance) test p value for weekdays and weekends log:", stats.levene(wkday_counts['comment_count_log'], wkend_counts['comment_count_log']).pvalue)
    print()
    
    """
    # Try exp (overflow encountered!!)
    
    wkday_counts['comment_count_exp'] = np.exp(wkday_counts['comment_count']) 
    wkend_counts['comment_count_exp'] = np.exp(wkend_counts['comment_count'])
    
    print("Normal test p value for weekdays comment counts exp:", stats.normaltest(wkday_counts['comment_count_exp']).pvalue) # Cannot reject
    print("Normal test p value for weekends comment counts exp:", stats.normaltest(wkend_counts['comment_count_exp']).pvalue) # Can reject
    print()
    """
    
    # Try square
    
    wkday_counts['comment_count_sqr'] = wkday_counts['comment_count'] ** 2
    wkend_counts['comment_count_sqr'] = wkend_counts['comment_count'] ** 2
    
    print("Normal test p value for weekdays comment counts squared:", stats.normaltest(wkday_counts['comment_count_sqr']).pvalue) # Cannot reject
    print("Normal test p value for weekends comment counts squared:", stats.normaltest(wkend_counts['comment_count_sqr']).pvalue) # Cannot reject
    print ("Levene (variance) test p value for weekdays and weekends squared:", stats.levene(wkday_counts['comment_count_sqr'], wkend_counts['comment_count_sqr']).pvalue)
    print()
    
    ### Fix 2: The Central Limit Theorm might save us. ###
    
    # The central limit theorem says that if our numbers are large enough, and we look at sample means,
    # then the result should be normal. Let's try that: we will combine all weekdays and weekend days 
    # from each year/week pair and take the mean of their (non-transformed) counts.
    
    # Get year / week / day and iso subreddit == canada
    wkday_counts['year'] = wkday_counts['date'].dt.isocalendar().year
    wkend_counts['year'] = wkend_counts['date'].dt.isocalendar().year
    wkday_counts['week'] = wkday_counts['date'].dt.isocalendar().week
    wkend_counts['week'] = wkend_counts['date'].dt.isocalendar().week
    
    # Perform groupby to extract weekdays and weekends by week / year
    wkday_counts_gpy = wkday_counts.groupby(['year','week']).mean()
    wkend_counts_gpy = wkend_counts.groupby(['year','week']).mean()
    
    # Check these agains normality and variance assumptions
    weekly_weekday_normality_p = stats.normaltest(wkday_counts_gpy['comment_count']).pvalue
    weekly_weekend_normality_p = stats.normaltest(wkend_counts_gpy['comment_count']).pvalue
    weekly_levene_p = stats.levene(wkday_counts_gpy['comment_count'], wkend_counts_gpy['comment_count']).pvalue
    print("Normal test p value for weekdays aggregated over entire data set:", stats.normaltest(wkday_counts_gpy['comment_count']).pvalue) # Cannot reject
    print("Normal test p value for weekends aggregated over entire data set:", stats.normaltest(wkend_counts_gpy['comment_count']).pvalue) # Cannot reject
    print ("Levene (variance) test p value for weekdays and weekends aggregated over entire data set:", stats.levene(wkday_counts_gpy['comment_count'], wkend_counts_gpy['comment_count']).pvalue)
    print()

    # Since these data sets now pass the normal and equivar tests, perform a ttest
    
    weekly_ttest = stats.ttest_ind(wkday_counts_gpy['comment_count'], wkend_counts_gpy['comment_count']).pvalue
    print("Ttest p-value for weekdays and weekends aggregated:", weekly_ttest)

    ### Fix 3: a non-parametric test might save us. ###
    # The Mann-Whitney U-test does not assume normally-distributed values.
    # Perform on the original, non-transformed, non-aggregated counts.
    # Two sided.
    
    mwu_test = stats.mannwhitneyu(wkday_counts['comment_count'], wkend_counts['comment_count'])
    print("Result from the Mann-Whitney U Test, p value:", mwu_test.pvalue) # p is small, so we can reject
    print()

    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p=ttest.pvalue,
        initial_weekday_normality_p=ogwkday_normality_pval,
        initial_weekend_normality_p=ogwkend_normality_pval,
        initial_levene_p=initial_levene,
        transformed_weekday_normality_p=transformed_weekday_normality_p,
        transformed_weekend_normality_p=transformed_weekend_normality_p,
        transformed_levene_p=transformed_levene_p,
        weekly_weekday_normality_p=weekly_weekday_normality_p,
        weekly_weekend_normality_p=weekly_weekend_normality_p,
        weekly_levene_p=weekly_levene_p,
        weekly_ttest_p=weekly_ttest,
        utest_p=mwu_test.pvalue,
    ))

if __name__ == '__main__':
    main()
