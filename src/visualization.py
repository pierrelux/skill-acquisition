#!/usr/bin/env python
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from itertools import *
from sklearn.cluster import spectral_clustering
import igraph as ig
import numpy as np
import argparse
import cPickle


parser = argparse.ArgumentParser(description='Plot the cluster graph')
parser.add_argument('dataset', help='Input dataset used to build the index')
parser.add_argument('graph', help='the state-space graph')
parser.add_argument('configuration', help='the configuration file')
parser.add_argument('clustering', nargs='?', help='the graph clustering')

args = parser.parse_args()

# Import dataset
print 'Loading dataset...'
dataset = np.loadtxt(args.dataset)

# Load graph
print 'Loading graph...'
graph = ig.load(args.graph)

# Parse domain configuration
obstacles = []
target_pos = []
target_rad = 0.01
ball_rad = 0.01
start_pos = []
with open(args.configuration) as fp:
    for line in fp.readlines():
        tokens = line.strip().split()
        if not len(tokens):
            continue
        elif tokens[0] == 'polygon':
            obstacles.append((zip(*[iter(map(float, tokens[1:]))] * 2)))
        elif tokens[0] == 'target':
            target_pos = [float(tokens[1]), float(tokens[2])]
            target_rad = float(tokens[3])
        elif tokens[0] == 'start':
            start_pos = zip(*[iter(map(float, tokens[1:]))]*2)
        elif tokens[0] == 'ball':
            ball_rad = float(tokens[1])

fig = plt.figure()
ax = fig.add_subplot(111)

ax.invert_yaxis()
ax.axis('equal')

for obs in obstacles:
    ax.add_patch(plt.Polygon(obs, fc='#404040'))

ax.add_patch(plt.Circle(target_pos, target_rad, ec='None', fc='red'))

# Draw graph
edges = LineCollection(([dataset[v1][:2], dataset[v2][:2]] for v1, v2 in graph.get_edgelist()))
edges.set_color('Lavender')
edges.set_zorder(1)
ax.add_collection(edges)

# Draw points
params = {}
if args.clustering:
    # Open the vertex dendogram
    print 'Loading clustering...'
    membership = cPickle.load(open(args.clustering, 'rb'))
    params['c'] = membership
    params['cmap'] = plt.get_cmap('hsv')

    # Highlight boundary points
    boundary = [v.index for v in graph.vs
            if sum((membership[nid] != membership[v.index]
                for nid in graph.neighbors(v.index))) > 7]

    print len(boundary)

ax.scatter(dataset[:,0], dataset[:,1], edgecolors='none', s=10, **params)

if args.clustering:
    ax.scatter(dataset[boundary,0], dataset[boundary,1], marker='o', s=12, facecolor='None')

#plt.show()
plt.savefig("neighborhood-graph.png", dpi=100)
