# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:06:15 2019

@author: DSU
"""

import numpy as np
import random
from itertools import accumulate
from bisect import bisect_left

def aco_iterative(locs, count=16, factor=0.1, decay=0.9):
    """
    Arguments:
        locs (atlas): atlas type object
        count (int): how many ants to use
        factor (float): pheromone factor for the ants must be > 0
        decay (float): decay factor for phermomone between 0. and 1.
    Yields:
        list: best path for this iteration
    """
    
    # residual pheromone and current iteration pheromone
    pheromone = np.full_like(locs.dist, factor)
    delta = np.empty_like(pheromone)
    
    while True:
        # best path and reset current iteration pheromone
        min_dist, min_path = float('inf'), None
        delta.fill(0.0)
        
        for _ in range(count):
            # current loc, path traveled, locs to vist yet
            cur, path, todo = 0, [0], list(range(1,len(locs)))
            
            # find cycle based on probability
            for _ in range(len(todo)):
                attraction =[pheromone[cur][i]/locs.dist[cur][i] for i in todo]
                total = sum(attraction)
                attraction = list(accumulate(x / total for x in attraction))
                cur = todo.pop(bisect_left(attraction, random.random()))
                path.append(cur)
            
            # update best path
            dist = locs.distance(path)
            if dist < min_dist:
                min_dist, min_path = dist, path
        
            # update delta pheromones
            for a, b in zip(path, path[1:] + [0]):
                delta[a][b] += factor / dist
        
        # update pheromones for decay, and symmetric delta pheromones
        pheromone = pheromone * decay + delta + delta.transpose()
        
        yield min_path

def aco_final(locs, args={}, show='', until='exp'):
    """
    Arguments:
        locs (atlas): atlas type object
        args (dict): args for the iterative tsp function call
        show (str): which paths to display
            '5': display best every (int) iterations
            'best': displays path if it is an improvement
            'all': displays path after each iteration
            'x': displays no paths
        until (str): 
            'x': how many iterations of ant colony to do
            'exp': stops when it's run for 2x time when it last improved at x
    Returns:
        list: the best path
    """
    min_dist, min_path = float('inf'), None
    iter_path = iter(aco_iterative(locs, **args))
    
    def iterate_display(i):
        nonlocal min_dist, min_path
        path = next(iter_path)
        dist = locs.distance(path)
        change = False
        
        if dist < min_dist:
            change = True
            min_dist, min_path = dist, path
            if show == 'best':
                locs.display(min_path,title=f'Gen: {i:3} - {min_dist:.1f}')
        if show == 'all':
            locs.display(path, title=f'Gen: {i:3} - {min_dist:.1f}')
        if show.isdigit() and i % int(show) is 0:
            locs.display(min_path, title=f'Gen: {i:3} - {min_dist:.1f}')
        
        return change
    
    if until == 'exp':
        last_update, i = 0, 0
        
        while 2 * last_update >= i or i < 100:
            i += 1
            if iterate_display(i):
                last_update = i
    elif until.isdigit():
        for i in range(int(until)):
            iterate_display(i)
            
    else:
        print('Invalid <until> specified.')
            
    
    return min_path
    
    
    
    



        
        
        
        
        
        
        