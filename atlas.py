# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:38:13 2019

@author: Nathan
"""

import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

class atlas:
    def __init__(self, **kwargs):
        """
        Arguments:
            file (str): csv file to read x,y values from
            lo (number): lower bound for x/y coordinates
            hi (number): upper bound for x/y coordinates
            n (int): how many locations to generate; must be >= 0
        Returns:
            (atlas): an atlas type object used for locations
        """
        if 'file' in kwargs:
            self.coords = np.genfromtxt(kwargs['file'], delimiter=',')
            self.lo, self.hi = self.coords.min(), self.coords.max()
            self.n = len(self.coords)
        else:
            # set parameters with defaults
            self.lo, self.hi = kwargs.get('lo', 0), kwargs.get('hi', 1000)
            self.n = kwargs.get('n', 32)
            # randomly generated x/y coordinates
            self.coords = np.random.uniform(self.lo, self.hi, (self.n, 2))
        
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