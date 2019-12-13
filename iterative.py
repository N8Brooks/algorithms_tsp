# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:06:15 2019

@author: Nathan
"""

import numpy as np
from itertools import accumulate, combinations
from bisect import bisect_left
import random
from random import sample, choices, randrange

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

def genetic(locs, select=3, size=100):
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

def two_opt(locs):
    """
    Arguments:
        locs (atlas): atlas type object
    Yields:
        list: the current path based on 2-opt heuristic
    """
    indexes = range(len(locs) + 1)
    path = min_path = list(range(len(locs)))
    dist = min_dist = locs.distance(path)
    while True:
        for i, j in combinations(indexes, r=2):
            yield path
            path = min_path[:i] + min_path[i:j][::-1] + min_path[j:]
            dist = locs.distance(path)
            if dist < min_dist:
                min_dist, min_path = dist, path

def three_opt(locs):
    """NOT IMPLEMENTED"""
    def all_segments(n: int):
        """Generate all segments combinations"""
        return ((i, j, k)
            for i in range(n)
            for j in range(i + 2, n)
            for k in range(j + 2, n + (i > 0)))
    
    def distance(i, j):
        return abs(j - i)
    
    def reverse_segment_if_better(tour, i, j, k):
        """If reversing tour[i:j] would make the tour shorter, then do it."""
        # Given tour [...A-B...C-D...E-F...]
        A, B, C, D, E, F = tour[i-1], tour[i], tour[j-1], tour[j], tour[k-1], tour[k % len(tour)]
        d0 = distance(A, B) + distance(C, D) + distance(E, F)
        d1 = distance(A, C) + distance(B, D) + distance(E, F)
        d2 = distance(A, B) + distance(C, E) + distance(D, F)
        d3 = distance(A, D) + distance(E, B) + distance(C, F)
        d4 = distance(F, B) + distance(C, D) + distance(E, A)
    
        if d0 > d1:
            tour[i:j] = reversed(tour[i:j])
            return -d0 + d1
        elif d0 > d2:
            tour[j:k] = reversed(tour[j:k])
            return -d0 + d2
        elif d0 > d4:
            tour[i:k] = reversed(tour[i:k])
            return -d0 + d4
        elif d0 > d3:
            tmp = tour[j:k] + tour[i:j]
            tour[i:k] = tmp
            return -d0 + d3
        return 0
    
    path = range(len(locs))
    while True:
        delta = 0
        for (a, b, c) in all_segments(len(locs)):
            delta += reverse_segment_if_better(locs, a, b, c)
        if delta >= 0:
            break
    return path

def aco_final(locs, args={}, show='', until='exp', func=genetic):
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
    
    
    
    



        
        
        
        
        
        
        