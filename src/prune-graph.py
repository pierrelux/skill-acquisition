#!/usr/bin/env python
from itertools import *
import igraph as ig
import numpy as np
import argparse
import cPickle
import os

def ccw(A,B,C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def intersect(A,B,C,D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def get_edges(points):
    a, b = tee(np.vstack([np.array(points), points[0]]))
    next(b, None)
    return izip(a, b)

def feasible_edge(p1, p2, obstacles):
    for obs in obstacles:
        for edge in get_edges(obs):
            if intersect(edge[0], edge[1], np.asarray(p1), np.asarray(p2)):
                return False
    return True

parser = argparse.ArgumentParser(description='Prune the proximity graph to remove impossible transitions.')
parser.add_argument('dataset', help='Input dataset used to build the index')
parser.add_argument('graph', help='the graph clustering')
parser.add_argument('configuration', help='the configuration file')

args = parser.parse_args()

graph_filename = os.path.splitext(os.path.basename(args.graph))[0]

# Import dataset
print 'Loading dataset...'
dataset = np.loadtxt(args.dataset)

# Load graph
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

impossible_edges = [(v1, v2) for v1, v2 in graph.get_edgelist()
                       if not feasible_edge(dataset[v1,:2], dataset[v2,:2], obstacles)]

print 'Edges deleted: ', len(impossible_edges)
graph.delete_edges(impossible_edges)

print 'Is connected ? ',  graph.is_connected()

graph.save(graph_filename + '.pickle')
