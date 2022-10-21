import sys
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy import stats


OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value:  {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value:  {more_searches_p:.3g} \n'
    '"Did more/less instructors use the search feature?" p-value:  {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value:  {more_instr_searches_p:.3g}'
)


def main():
    searchdata_file = sys.argv[1]
    df = pd.read_json(searchdata_file, orient='records', lines=True)
    
    """
    
    **Users with odd-numbered uid were shown a new and improved search box**
    
    Questions to answer: Did users shown the new design use it more? 
        
            1. Did a different fraction of users have search count > 0?
            2. Is the number of searches per user different?
    
    Note: Use nonparametric testing
    Get p-values for the above questions
    
    """
    # Tackle Question 1 for all users.
    # Split data into new and improved and original based on uid
    
    new = df[df['uid'] % 2 == 1].copy()
    og = df[df['uid'] % 2 == 0].copy()
    
    # contingency table
    
    ct = pd.DataFrame(index=['New', 'Old'], columns=['0', '> 0'])
    ct['0'][0] = len(new[new['search_count'] == 0])
    ct['> 0'][0] = len(new[new['search_count'] > 0])
    ct['0'][1] = len(og[og['search_count'] == 0])
    ct['> 0'][1] = len(og[og['search_count'] > 0])
    
    t_users, p_users, dof_users, freq_users = chi2_contingency(ct) 
    
    # Tackle Question 2 for all users.
    # Null hypothesis will be that the sum of the rankings in the two groups does not differ
    
    mwu_searches = stats.mannwhitneyu(new['search_count'], og['search_count'])
    
    # Tackle Question 1 for all instructors.
    # contingency table
    
    ct2 = pd.DataFrame(index=['New', 'Old'], columns=['0', '> 0'])
    ct2['0'][0] = len(new[ (new['search_count'] == 0) & (new['is_instructor'] == True)])
    ct2['> 0'][0] = len(new[ (new['search_count'] > 0) & (new['is_instructor'] == True)])
    ct2['0'][1] = len(og[ (og['search_count'] == 0) & (og['is_instructor'] == True)])
    ct2['> 0'][1] = len(og[ (og['search_count'] > 0) & (og['is_instructor'] == True)])
    
    t_inst, p_inst, dof_inst, freq_inst = chi2_contingency(ct2) 
    
    # Tackle Question 2 for all instructors.
    # Null hypothesis will be that the sum of the rankings in the two groups does not differ
    
    new_instr = df[(df['uid'] % 2 == 1) & (df['is_instructor'] == True)].copy()
    og_instr = df[(df['uid'] % 2 == 0)  & (df['is_instructor'] == True)].copy()
    
    mwu_instr_searches = stats.mannwhitneyu(new_instr['search_count'], og_instr['search_count'])
    
    # Output
    print(OUTPUT_TEMPLATE.format(
        more_users_p=p_users,
        more_searches_p=mwu_searches.pvalue,
        more_instr_p=p_inst,
        more_instr_searches_p=mwu_instr_searches.pvalue,
    ))


if __name__ == '__main__':
    main()
