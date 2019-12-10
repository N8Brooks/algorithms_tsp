# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:26:42 2019

@author: DSU
"""

from itertools import permutations, islice
from math import factorial as f

def brute(locs):
    """
    Aruments:
        locs (atlas): an atlas type object
    Returns:
        (list): an order describing the brute force based tsp path found
    """
    n = len(locs)
    return min(islice(permutations(range(n), n), f(n)//2), key=locs.distance)

