# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:07:42 2019

@author: Nathan
"""

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.optimize import linear_sum_assignment
import sys

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

def christofide(locs):
    """
    Aruments:
        locs (atlas): an atlas type object
    Returns:
        (list): a path found using the christofide tsp method
    """
    
    # compute minimum spanning tree
    mst = minimum_spanning_tree(locs.dist).toarray().astype(float)
    
    # find nodes with odd degree
    odd = np.count_nonzero(mst, axis=0) + np.count_nonzero(mst, axis=1)
    odd = (odd - (mst.diagonal() > 0)) % 2
    
    # subgraph using nodes with odd degree
    sub = np.outer(odd, odd) * locs.dist
    nonzero = np.nonzero(odd)[0]
    zero = np.where(odd == 0)[0]
    sub = np.delete(sub, zero, 0)
    sub = np.delete(sub, zero, 1)
    np.fill_diagonal(sub, sys.float_info.max)  # inf
    
    # compute minimum weight perfect matching
    order = dict(enumerate(nonzero))
    mapping = np.vectorize(lambda x: order[x])
    coords = mapping(linear_sum_assignment(sub))
    
    # union of mst and subgraph
    for r, c in coords.transpose():
        mst[c,r] = mst[r,c] = locs.dist[r,c]
    
    # find euler tour
    path = list()
    def parse(cur):
        for x in range(len(mst)):
            if mst[cur,x] > 0 or mst[x,cur] > 0:
                mst[cur][x] = mst[x][cur] = 0
                parse(x)
        path.append(cur)
    parse(0)
    
    # return hamiltonian circuit by short circuiting
    return [x for i, x in enumerate(path) if x not in path[:i]]
    
    
    
    
    
    