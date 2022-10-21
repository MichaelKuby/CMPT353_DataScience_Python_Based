#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 09:14:06 2022

@author: michaelkuby
"""

import sys
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def main():
    datafile = sys.argv[1]
    df = pd.read_csv(datafile)

    """    
    We have multiple groups, and we want to determine if the means of any of
    the groups differ. We can use the ANOVA test if the assumptions are met:
        
        1. Observations are independent and identically distributed (true)
        2. Groups are normally distributed (True because of the CLT)
        3. Groups have equal variance (probably true)?
    """        
    anova_F, anova_P = stats.f_oneway(df['qs1_elapsed'], df['qs2_elapsed'], df['qs3_elapsed'], df['qs4_elapsed'], df['qs5_elapsed'], df['merge_elapsed'], df['partition_elapsed'])
    
    print("P value from the one way anova test: ",  anova_P)

    print("\nIf", anova_P, " < .05 we may conclude that there a difference in the means of some of the groups,\n" 
          "but to find out which we must do some post hoc analysis.")
    
    # Plan: Use Tukey's Honest Significance Difference Test
    # Get the data in the proper format
    
    melted_df = df.melt()

    posthoc = pairwise_tukeyhsd(
        melted_df['value'], melted_df['variable'],
        alpha = 0.05)
    
    print(posthoc)

    fig = posthoc.plot_simultaneous()
    
    """
    We see that we can reject the null for:
        merge vs. qs1
        merge vs. qs2
        merge vs. qs3
        
        partition vs qs1-5
        
        qs1 vs qs4
        qs1 vs qs5

        qs2 vs qs4
        qs2 vs qs5
        
        qs3 vs qs4
        qs3 vs qs5
        
    We should now do a individual one-sided tests on each.        
    """
    
    merge_qs1 = stats.ttest_ind(df['merge_elapsed'], df['qs1_elapsed'])
    merge_qs2 = stats.ttest_ind(df['merge_elapsed'], df['qs2_elapsed'])
    merge_qs3 = stats.ttest_ind(df['merge_elapsed'], df['qs3_elapsed'])
    
    partition_qs1 = stats.ttest_ind(df['partition_elapsed'], df['qs1_elapsed'])
    partition_qs2 = stats.ttest_ind(df['partition_elapsed'], df['qs2_elapsed'])
    partition_qs3 = stats.ttest_ind(df['partition_elapsed'], df['qs3_elapsed'])
    
    qs1_qs2 = stats.ttest_ind(df['qs1_elapsed'], df['qs2_elapsed'])
    qs1_qs4 = stats.ttest_ind(df['qs1_elapsed'], df['qs4_elapsed'])
    qs1_qs5 = stats.ttest_ind(df['qs1_elapsed'], df['qs5_elapsed'])
    
    qs2_qs4 = stats.ttest_ind(df['qs2_elapsed'], df['qs4_elapsed'])
    qs2_qs5 = stats.ttest_ind(df['qs2_elapsed'], df['qs5_elapsed'])
    
    qs3_qs4 = stats.ttest_ind(df['qs3_elapsed'], df['qs4_elapsed'])
    qs3_qs5 = stats.ttest_ind(df['qs3_elapsed'], df['qs5_elapsed'])
    
    print()
    print("Merge sort vs. QS1: ", merge_qs1.pvalue)
    print("Merge sort vs. QS2: ", merge_qs2.pvalue)
    print("Merge sort vs. QS3: ", merge_qs3.pvalue)
    print()
    print("Partition sort vs. QS1: ", partition_qs1.pvalue)
    print("Partition sort vs. QS2: ", partition_qs2.pvalue)
    print("Partition sort vs. QS3: ", partition_qs3.pvalue)
    print()
    print("QS1 vs QS2: ", qs1_qs2.pvalue)
    print("QS1 vs QS4: ", qs1_qs4.pvalue)
    print("QS1 vs QS5: ", qs1_qs5.pvalue)
    print()
    print("QS2 vs QS4: ", qs2_qs4.pvalue)
    print("QS2 vs QS4: ", qs2_qs5.pvalue)
    print()    
    print("QS3 vs QS4: ", qs3_qs4.pvalue)
    print("QS3 vs QS4: ", qs3_qs5.pvalue)
    
    
    
if __name__ == '__main__':
    main()