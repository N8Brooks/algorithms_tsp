# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:38:13 2019

@author: Nathan
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.spatial import distance
from collections.abc import Iterator
from itertools import islice
import argparse
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
            xd = np.cos(theta[0])-np.cos(theta[1])
            yd = np.sin(theta[0])-np.sin(theta[1])
            random.shuffle(theta)
            c, r = (lo + hi) / 2, hi - (lo + hi) / 2
            print(f'Optimal: {r * np.hypot(xd, yd) * self.n:.1f}')
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
    
    def compute(self, func, show='best', ret='', save='', until=100, **kwargs):        
        """
        Arguments:
            func (function): tsp function to compute
            until (int): how many iterations to use if iterable
            show (str): specify what to display
                'improve' = show any path that improves
                'all' = show path after each call
                'best' = show best path after all iterations
                '' or 'none' = display nothing
            ret (str): specify what to return
                'dist' = distance of minimum path
                'path' = the minimum path found
                '' or 'none' = None
            save (str): Setting means mp4 will be saved of images titled (str)
                'file': would save images as 'file.mp4'
                '' or 'none': nothing is saved to disk
            kwargs: what arguments to pass to tsp function
        Returns:
            list, float, or None: depends on what ret is - default is None
        """
        # display and save paths if specified to save
        paths = list()
        def display(path, dist, i=None):
            xs, ys = zip(*map(lambda x: self.coords[x], path + [path[0]]))
            plt.clf()
            plt.plot(xs, ys, 'ro-')
            title = f'{dist:.1f}' if i is None else f'Gen: {i:3} - {dist:.1f}'
            plt.title(title)
            if save and save != 'none': paths.append((xs, ys, title))
            plt.pause(0.1)
        
        plt.ion()
        # it is an iterative function
        min_path, min_dist = None, float('inf')
        if isinstance(func(self), Iterator):
            min_i = 0
            for i, path in enumerate(islice(func(self, **kwargs), until)):
                dist = self.distance(path)
                if dist < min_dist:
                    min_dist, min_path, min_i = dist, path, i
                    if show == 'improve':
                        display(path, dist, i)
                if show == 'all':
                    display(path, dist, i)
            if show == 'best':
                display(min_path, min_dist, min_i)
            if show and show != 'none':
                plt.show()
        # it is not an iterative function
        else:
            min_path = func(self, **kwargs)
            min_dist = self.distance(min_path)
            if show and show != 'none':
                self.display(min_path, min_dist)
                plt.show(block=True)
        
        # save as an mp4 with save variable as title
        if save and save != 'none':
            def animate(i):
                plt.clf()
                plt.title(paths[i][2])
                return plt.plot(*paths[i][:2], 'ro-')
            
            anim = animation.FuncAnimation(plt.gcf(),animate,frames=len(paths))
            anim.save(save+'.mp4')

        # return specified type
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

if __name__ == '__main__':
    import iterative, heuristic, exact
    parser = argparse.ArgumentParser(description=('Traveling Salesman algorith'
                                                  'm computation and display'))
        
    # compute parameters
    parser.add_argument('--algorithm', metavar='-a', type=str, default=('sim_a'
                        'nnealing'), help=('name of traveling salesman algorit'
                        'hm to use'))
    parser.add_argument('--show', metavar='-s', type=str, default='improve', help=('What to display while computing:\nbest: only the best path found\nimprove: display any improvement\nall: display every iteration\nnone: display nothing'))
    parser.add_argument('--ret', metavar='-r', type=str, default='none', help=('what to return as output:\ndist: prints lowest distance found\npath: prints order of lowest path\n:none: prints nothing'))
    parser.add_argument('--save', metavar='-S', type=str, default='none', help=('if not "none", it will save as <save var>.mp4'))
    parser.add_argument('--until', metavar='-u', type=int, default=100, help=('how many iterations to do (useless for non iterative algorithms)'))
    
    # atlas parameters
    parser.add_argument('--type', metavar='-t', type=str, default='r', help=('what locations to use:\n<file_name.csv>: csv file to read location x,y pairs from\nr: random locations\n c: generate a circle of locations'))
    parser.add_argument('--count', metavar='-c', type=int, default=32, help=('how many locations to generate (ignored if reading from a file)'))
    parser.add_argument('--lo', metavar='-l', type=float, default=0., help=('lower limit for x and y when generating locations (ignored if reading from a file'))
    parser.add_argument('--hi', metavar='-u', type=float, default=1000., help=('upper limit for x and y when generating locations (ignored if reading from a file'))

    args = parser.parse_args()

    # look for algorithm in other files
    algo = getattr(iterative, args.algorithm, None)
    if algo is None: algo = getattr(heuristic, args.algorithm, None)
    if algo is None: algo = getattr(exact, args.algorithm, None)
    if algo is None: raise ValueError("Algorithm was not found.")
    
    # generate or read locations
    if args.type == 'r': locs = atlas(args.count, args.lo, args.hi)
    elif args.type == 'c': locs = atlas(str(args.count), args.lo, args.hi)
    else: locs = atlas(args.type)
    
    # run algorithm
    locs.compute(algo, show=args.show, ret=args.ret, \
                 save=args.save, until=args.until)
    
    
    
    
    
    
    
    
    
    
    