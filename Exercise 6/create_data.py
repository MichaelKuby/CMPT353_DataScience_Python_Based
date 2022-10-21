#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 18:47:31 2022

@author: MichaelKuby
"""

import time
import numpy as np
import pandas as pd
from implementations import all_implementations

"""
Generate arrays with random integers, randomly sorted
As large as possible to assume normality
"""
def main():
    # Central Limit Theorem suggests if n > 30 we can begin to assume normality
    np.random.seed(0)
    rands = 50
    rand = pd.DataFrame(np.random.choice(200, (50, rands)))
    
    df = pd.DataFrame(index = range(rands), columns=['qs1_elapsed', 'qs2_elapsed', 'qs3_elapsed', 'qs4_elapsed', 'qs5_elapsed', 'merge_elapsed', 'partition_elapsed'])
    
    i = 0
    for columnName in rand:
        j = 0
        for sort in all_implementations:
            st = time.time()
            res = sort(rand[columnName])
            en = time.time()
            df.iloc[i][j] = en - st
            j+=1
        i+=1
    
    df.to_csv('data.csv', index=False)

if __name__ == '__main__':
    main()