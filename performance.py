# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:57:37 2019

@author: DSU
"""

import pandas as pd
import exact
from atlas import atlas
from time import clock
from tqdm import trange

algorithms = [exact.brute, exact.recursive, exact.dynamic]

# compute all algorithms as far as I can
df = pd.DataFrame()
for i in trange(20):
    record = pd.Series(name=i)
    
    locs = atlas(i)
    
    for algo in algorithms:
        start = clock()
        locs.compute(algo, ret='', show='')
        record[algo.__name__] = clock() - start
    
    df = df.append(record)

# compute dynamic algorithm as far as I can
algo = exact.dynamic
for i in trange(14,30):
    record = pd.Series(name=i)
    
    locs = atlas(i)
    
    time = clock()
    locs.compute(algo, ret='', show='')
    record[algo.__name__] = clock() - start
    
    df = df.append(record)
    