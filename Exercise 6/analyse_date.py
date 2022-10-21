#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 09:14:06 2022

@author: michaelkuby
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

datafile = sys.argv[1]
df = pd.read_csv(datafile)

