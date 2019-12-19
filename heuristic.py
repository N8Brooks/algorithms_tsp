# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:07:42 2019

@author: Nathan
"""

from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.optimize import linear_sum_assignment
from collections import Counter
import numpy as np
import sys

def greedy(locs, start=0):
    """
    Aruments:
        locs (atlas): an atlas type object
        start (int): the location to begin nearest neighbors at
    Returns:
        (list): a list of locations describing the greedy based tsp path found
    """
    # make a copy of their distances for editing
    remaining_dist = locs.dist.copy()
    # the path build by nearest neighbors
    path = [start]
    
    for _ in range(1, len(locs)):
        # remove the distances to previous city so they aren't chosen again
        remaining_dist[:,start] = np.inf
        # index of shortest distance to city not reached yet
        start = np.argmin(remaining_dist[start])
        # add to path shortest available distance to path
        path.append(start)
    
    return path

def mst(locs):
    """
    Aruments:
        locs (atlas): an atlas type object
    Returns:
        (list): a path found using the preorder mst tsp method
    """
    # compute minimum spanning tree
    mst = minimum_spanning_tree(locs.dist).toarray()
    mst += mst.transpose()
    
    # preorder traversal
    path = list()
    def parse(x):
        path.append(x)
        for y in np.nonzero(mst[x])[0]:
            if y not in path:
                parse(y)
    parse(0)
    
    return path
        

def christofide(locs, nn=False):
    """
    Aruments:
        locs (atlas): an atlas type object
        nn (bool): Use nearest neighbors instead of short circuiting
    Returns:
        (list): a path found using the christofide tsp method
    """
    # compute minimum spanning tree
    mst = minimum_spanning_tree(locs.dist).toarray()
    
    # find nodes with odd degree - mst has no self connecting nodes
    odd = np.count_nonzero(mst, axis=0) + np.count_nonzero(mst, axis=1)
    
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
    
    if nn:
        # difference of length with index i and without index i
        def step(i):
            a = path[i-1] if i != 0 else path[len(path)-1]
            b = path[i]
            c = path[i+1] if i != len(path) - 1 else path[0]
            return locs.dist[a,b] + locs.dist[b,c] - locs.dist[b,c]
        
        # keep only one of each location by removing non smallest differences
        extra = Counter(path)
        for key, val in extra.items():
            if val < 2: continue
            within = [(step(i),i) for i, x in enumerate(path) if x == key]
            within = min(within, key=lambda x: x[0])[1]
            path = [x for x in path[:within] if x != key] + [key] + \
                    [x for x in path[within+1:] if x != key]
    else:
        # hamiltonian circuit by cutting off
        path = [x for i, x in enumerate(path) if x not in path[:i]]
    
    return path

    
    
    
    
    