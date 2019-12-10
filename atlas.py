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
        self.lo, self.hi, self.n = lo, hi, n
        self.coords = np.random.uniform(lo, hi, (n, 2))
        self.dist = distance.cdist(self.coords, self.coords)
    
    def __len__(self):
        return self.n
    
    def display(self, path, title=None):
        x1, y1 = self.coords[path[-1]]
        for x2, y2 in map(lambda x: self.coords[x], path):
            plt.plot((x1,x2), (y1,y2), 'ro-')
            x1, y1 = x2, y2
        plt.title(title if title else f'{self.distance(path):.3f}')
        plt.show()
    
    def distance(self, path):
        return sum(self.dist[a][b] for a,b in zip(path, path[1:])) + \
                      self.dist[path[-1]][path[0]]