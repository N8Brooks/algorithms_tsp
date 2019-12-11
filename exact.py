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
        (tuple): tsp path based on O(n!) brute force search
    """
    n = len(locs)
    if n < 2: return [0] if n is 1 else []
    return min(islice(permutations(range(n), n), f(n)//2), key=locs.distance)

def recursive(locs):
    """
    Aruments:
        locs (atlas): an atlas type object
    Returns:
        (list): tsp path based on O(n!) recursive brute force search
    """
    def worker(i, s):
        if not s:
            return locs.dist[i][0], [0]
        
        min_path, min_distance = None, float('inf')
        for x, j in enumerate(s):
            distance, path = worker(x, s[:x] + s[x+1:])
            distance += locs.dist[i][j]
            if distance < min_distance:
                min_distance, min_path = distance, path + [j]
        
        return min_distance, min_path

    return list() if len(locs) < 1 else worker(0, list(range(1, len(locs))))[1]

def dynamic(locs):
    """
    Aruments:
        locs (atlas): an atlas type object
    Returns:
        (list): tsp path based on O(n^2 2^n) dynamic programming approach
    """
    dp = dict()
    
    def worker(i, s):
        if s == (1 << len(locs)) - 1:
            return locs.dist[i][0], [0]
        else:
            if (i, s) in dp:
                return dp[(i, s)]
            
            min_distance, min_path = float('inf'), None
            for j in range(len(locs)):
                if s & (1 << j):
                    continue
                distance, path = worker(j, s | (1 << j))
                distance += locs.dist[i][j]
                if distance < min_distance:
                    min_distance, min_path = distance, path + [j]
            
            # return dp[(i, x)] := min_distance, min_path # python 3.8
            dp[(i, s)] = min_distance, min_path
            return min_distance, min_path
    
    return list() if len(locs) < 1 else worker(0, 1)[1]

# list of tsp algorithms in this script
algorithms = [brute, recursive, dynamic]

if __name__ == '__main__':
    """
    driver code for this scipt which verifies correctness
    generates several random locations and compares the algorithm's distances
    verifies coorectness by asserting they are within 1e-8 of each other
    """
    from atlas import atlas
    
    for i in range(10):
        locs = atlas(0, 1000, i)
        distances = [locs.distance(algo(locs)) for algo in algorithms]
        assert all((distances[0] - x) < 1e-8 for x in distances[1:])
    
    
    
    
    
    
    
    
    