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
    rng = np.random.default_rng()
    size_array = 100 # in the write up I used 100
    num_arrays = 50 # in the write up I used 50
    
    df = pd.DataFrame(index = range(num_arrays), columns=['qs1_elapsed', 'qs2_elapsed', 'qs3_elapsed', 'qs4_elapsed', 'qs5_elapsed', 'merge_elapsed', 'partition_elapsed'])
    
    for i in range(num_arrays):
        j = 0
        array = rng.integers(low=0, high=200, size=size_array)
        for sort in all_implementations:
            st = time.time()
            res = sort(array)
            en = time.time()
            df.iloc[i][j] = en - st
            j+=1
        i+=1
    
    df.to_csv('data.csv', index=False)

if __name__ == '__main__':
    main()