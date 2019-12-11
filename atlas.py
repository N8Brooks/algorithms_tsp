# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:38:13 2019

@author: Nathan
"""

import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

class atlas:
    def __init__(self, lo, hi, n):
        """
        Arguments:
            lo (number): lower bound for x/y coordinates
            hi (number): upper bound for x/y coordinates
            n (int): how many locations to generate; must be >= 0
        Returns:
            (atlas): an atlas type object used for locations
        """
        # parameters used for initialization
        self.lo, self.hi, self.n = lo, hi, n
        # randomly generated x/y coordinates
        self.coords = np.random.uniform(lo, hi, (n, 2))
        # n x n distance matrix
        self.dist = distance.cdist(self.coords, self.coords)
    
    def __len__(self):
        return self.n
    
    def display(self, path, title=None):
        x1, y1 = self.coords[path[-1]]
        for x2, y2 in map(lambda x: self.coords[x], path):
            plt.plot((x1,x2), (y1,y2), 'ro-')
            x1, y1 = x2, y2
        plt.title(title if title else f'{self.distance(path):.1f}')
        plt.show()
    
    def distance(self, path):
        if self.n is 0: return 0
        return sum(self.dist[a][b] for a,b in zip(path, path[1:])) + \
                      self.dist[path[-1]][path[0]]