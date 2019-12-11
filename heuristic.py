# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:07:42 2019

@author: Nathan
"""

import numpy as np

def greedy(locs):
    """
    Aruments:
        locs (atlas): an atlas type object
    Returns:
        (list): a list of locations describing the greedy based tsp path found
    """
    # create a memo of each locations number
    remaining_locs = list(range(1, len(locs)))
    # make a copy of their distances for editing
    remaining_dist = locs.dist.copy()
    # set each city's distance with itself to inf so it is not chosen
    np.fill_diagonal(remaining_dist, np.inf)
    # current city, current path, next city
    target, path, connect = 0, [0], None
    
    while remaining_locs:
        # index of shortest distance to city not reached yet
        connect = np.argmin(remaining_dist[target])
        # remove the distances to previous city so they aren't chosen again
        remaining_dist = np.delete(remaining_dist, target, 0)
        remaining_dist = np.delete(remaining_dist, target, 1)
        # index of chosen city in remaining cities
        target = connect if connect < target else connect - 1
        # add its location number to path and remove
        path.append(remaining_locs.pop(target))
    
    return path

