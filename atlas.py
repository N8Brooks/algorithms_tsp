# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:38:13 2019

@author: Nathan
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import distance
from collections.abc import Iterator
from itertools import islice
from iterative import *
from heuristic import *
from exact import *

class atlas:
    def __init__(self, create, lo=0, hi=1000):
        """
        Arguments:
            create (int): how many locations to randomly generate
                   (str): csv file to read x,y values from
                   (digit): a digit str will create a circle with create points
            lo (number): lower bound for x/y coordinates
            hi (number): upper bound for x/y coordinates
        Returns:
            (atlas): an atlas type object used for locations
        """
        if isinstance(create, str) and create.isdigit():
            # create int(create) number of points on circle
            self.n, self.lo, self.hi = int(create), lo, hi
            theta = [t for t in np.arange(0, 2 * np.pi, 2 * np.pi / self.n)]
            random.shuffle(theta)
            c, r = (lo + hi) / 2, hi - (lo + hi) / 2
            self.coords = c+r*np.array([(np.sin(t),np.cos(t)) for t in theta])
        elif isinstance(create, str):
            # read from file
            self.coords = np.genfromtxt(create, delimiter=',')
            self.lo, self.hi = self.coords.min(), self.coords.max()
            self.n = len(self.coords)
        else:
            # set variables
            self.n, self.lo, self.hi = create, lo, hi
            # randomly generated x/y coordinates
            self.coords = np.random.uniform(lo, hi, (self.n, 2))
        
        # n x n distance matrix
        self.dist = distance.cdist(self.coords, self.coords)
    
    def __len__(self):
        """
        Returns:
            int: count of locations
        """
        return self.n
    
    def save(self, file: str):
        """
        Arguments:
            file (str): csv file to save x,y locations to
        """
        np.savetxt(file, self.coords, delimiter=",")
    
    def display(self, path, title=None):
        """
        Arguments:
            path (list): order of path to show
            title (str): specify the title for displayed plot
        """
        x1, y1 = self.coords[path[-1]]
        for x2, y2 in map(lambda x: self.coords[x], path):
            plt.plot((x1,x2), (y1,y2), 'ro-')
            x1, y1 = x2, y2
        plt.title(title if title else f'{self.distance(path):.1f}')
        plt.show()
    
    def compute(self, func, show='best', ret='', until=100, **kwargs):        
        """
        Arguments:
            func (function): tsp function to compute
            show (str): specify what to display
                'improve' = show any path that improves
                'all' = show path after each call
                'best' = show best path after all iterations
                '' = display nothing
            ret (str): specify what to return
                'dist' = distance of minimum path
                'path' = the minimum path found
                '' = return None
            until (int): how many iterations to use if iterable
            kwargs: what arguments to pass to tsp function
        Returns:
            list, float, or None: depends on what ret is - default is None
        """
        min_path, min_dist = None, float('inf')
        # it is an iterative function
        if isinstance(func(self), Iterator):
            min_i = 0
            for i, path in enumerate(islice(func(self, **kwargs), until)):
                dist = self.distance(path)
                if dist < min_dist:
                    min_dist, min_path, min_i = dist, path, i
                    if show == 'improve':
                        self.display(path,title=f'Gen: {i:3} - {min_dist:.1f}')
                if show == 'all':
                    self.display(path,title=f'Gen: {i:3} - {min_dist:.1f}')
            if show == 'best':
                self.display(min_path,title=f'Gen: {min_i:3} - {min_dist:.1f}')
        # it is not an iterative function
        else:
            min_path = func(self, **kwargs)
            min_dist = self.distance(min_path)
            if show:
                self.display(min_path)
        
        if ret == 'path':
            return min_path
        elif ret == 'dist':
            return min_dist
    
    def distance(self, path):
        """
        Arguments:
            path (list): order of path to find distance of
        Returns:
            float: length of path
        """
        if self.n == 0: return 0
        assert self.n == len(path)
        return sum(self.dist[a][b] for a,b in zip(path, path[1:])) + \
                      self.dist[path[-1]][path[0]]