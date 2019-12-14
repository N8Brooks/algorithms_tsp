# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:06:15 2019

@author: Nathan
"""



import numpy as np
import random

from random import sample, choices, randrange
from heuristic import greedy
from itertools import accumulate, combinations
from bisect import bisect_left

def aco(locs, count=8, factor=1.0, decay=0.66):
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
    pheromone = np.ones_like(locs.dist)
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

def genetic(locs, select=33, size=100):
    """
    Arguments:
        locs (atlas): atlas type object
        select (int): how many members to keep after each generation
        size (int): the size of the population in each generation
    Yields:
        list: path found using gentic algorithm with SCX breeding
    """
    # intial random paths
    pop = [sample(range(len(locs)), len(locs)) for _ in range(size)]
    length = len(locs)
    
    def breed(par_a, par_b):
        # tsp path, last used node, set of not used locations
        last = randrange(length)
        path, legit = [last], set(range(length)) - {last}
        
        for i in range(length - 1):
            # find shorter first legitamate in either parent
            a = next((x for x in par_a[i:] if x in legit), *sample(legit, 1))
            b = next((x for x in par_b[i:] if x in legit), *sample(legit, 1))
            last = a if locs.dist[last][a] < locs.dist[last][b] else b
            
            # add to path and remove from legitamate
            legit.remove(last)
            path.append(last)
        
        # mutate
        a, b = sample(range(length), 2)
        path[a], path[b] = path[b], path[a]
        
        return path

    while True:
        # sort by fitness and yield best result
        pop.sort(key=locs.distance)
        yield pop[0]
        # breed those selected
        pop = [breed(*choices(pop[:select], k=2)) for _ in range(size)]

def two_opt(locs, nn=False):
    """
    Arguments:
        locs (atlas): atlas type object
        nn (bool): initialize path as nearest neighbor greedy path
    Yields:
        list: the current path based on 2-opt heuristic
    """
    indexes = range(len(locs))
    path = greedy(locs) if nn else list(indexes)
    d = lambda x, y: locs.dist[path[x], path[y]]
    while True:
        # generate all substrings of path
        for i, j in combinations(indexes, 2):
            yield path
            # flip substring if it shortens path
            if d(i-1, i) + d(j-1,j) > d(i-1,j-1) + d(i, j):
                path[i:j] = reversed(path[i:j])

def three_opt(locs, nn=False):
    """
    Arguments:
        locs (atlas): atlas type object
        nn (bool): initialize path as nearest neighbor greedy path
    Yields:
        list: the current path based on 3-opt heuristic
    """
    indexes = range(len(locs))
    path = greedy(locs) if nn else list(indexes)
    distance = lambda x, y: locs.dist[path[x], path[y]]
    while True:
        # generate all two adjacent substrings of path
        for i, j, k in combinations(indexes, 3):
            yield path
            dist = distance(i-1,i) + distance(j-1,j) + distance(k-1,k)
        
            # modify those substrings to be shortest
            if dist > distance(i-1, j-1) + distance(i, j) + distance(k-1, k):
                path[i:j] = reversed(path[i:j])
            elif dist > distance(i-1, i) + distance(j-1, k-1) + distance(j, k):
                path[j:k] = reversed(path[j:k])
            elif dist > distance(k, i) + distance(j-1, j) + distance(k-1, i-1):
                path[i:k] = reversed(path[i:k])
            elif dist > distance(i-1, j) + distance(k-1, i) + distance(j-1, k):
                path[i:k] = path[j:k] + path[i:j]

def iterate(locs, args={}, show='', until='exp', func=genetic):
    """
    Arguments:
        locs (atlas): atlas type object
        args (dict): args for the iterative tsp function call
        show (str): which paths to display
            '5': display best every (int) iterations
            'improve': displays path if it is an improvement
            'best': displays the best generation of all generated
            'all': displays path after each iteration
            'x': displays no paths
        until (str): 
            'x': how many iterations of ant colony to do
            'exp': stops when it's run for 2x time when it last improved at x
    Returns:
        list: the best path
    """
    min_dist, min_path = float('inf'), None
    iter_path = iter(func(locs, **args))
    
    def iterate_display(i):
        nonlocal min_dist, min_path
        path = next(iter_path)
        dist = locs.distance(path)
        change = False
        
        if dist < min_dist:
            change = True
            min_dist, min_path = dist, path
            if show == 'improve':
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
            
    if show == 'best':
        locs.display(min_path,title=f'Gen: {i:3} - {min_dist:.1f}')
    
    return min_path
    
    
    
    



        
        
        
        
        
        
        