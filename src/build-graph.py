#!/usr/bin/python

import sys
import random
import argparse
import itertools
import numpy as np
import igraph as ig
from pyflann import *


# Parse arguments
parser = argparse.ArgumentParser(description='Build the state-space graph')

parser.add_argument('dataset')
parser.add_argument('-s', '--sigma', action='store', type=float, default=0.5, dest='sigma', help='sigma value (default: 0.5)')
parser.add_argument('-p', '--prefix', action='store', type=str, dest='prefix', help="output prefix (default: dataset)")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--directed', metavar='KNN', type=int, help="build a directed knn neighborhood graph")
group.add_argument('--mutual', metavar='KNN', type=int, help="build a mutual knn neighborhood graph")
group.add_argument('--symmetric', metavar='KNN', type=int, help="build a symmetric knn neighborhood graph")
group.add_argument('--radius', metavar='RADIUS', type=float, help="build a r-neighborhood graph")

args = parser.parse_args()

if not args.prefix:
    args.prefix = os.path.splitext(os.path.basename(args.dataset))[0]

# Load csv file
dataset = np.loadtxt(sys.argv[1])

# Compute nearest neighbors
print 'Building kd-tree index...'
flann = FLANN()
flann.build_index(dataset)

# Create the state-space graph
graph = ig.Graph(directed=args.directed)
graph.add_vertices(np.alen(dataset))

if args.radius:
    print 'Searching all nearest neighbors within r=%f...'%(args.radius,)
    edges = []
    weights = []
    for i in xrange(np.alen(dataset)):
        nn, dists = flann.nn_radius(dataset[i,:], args.radius)
        edges.extend(((i, j) for j in nn))
        weights.extend((np.exp(-1*d/args.sigma) for d in dists))

    graph.add_edges(edges)
    graph.es["weight"] = weights
else:
    k = next((n for n in [args.directed, args.mutual, args.symmetric] if n))

    print 'Searching all %d nearest neighbors in kd-tree...'%(k,)

    knn, dists = flann.nn_index(dataset, k)
    knn = knn[:, 1:]
    dists = dists[:, 1:]

    if args.directed:
        print 'Building directed graph...'
        graph.add_edges(((i, j) for i in xrange(np.alen(knn)) for j in knn[i,:]))

    if args.mutual:
        print 'Building mutual graph...'
        graph.add_edges(((i, j) for i in xrange(np.alen(knn)) for j in knn[i,:] if i in knn[j,:]))

    if args.symmetric:
        print 'Building symmetric graph...'
        complete_edges = (((i, j) for i in xrange(np.alen(knn)) for j in knn[i,:]))
        graph.add_edges(map(list, list(set(map(frozenset, complete_edges)))))

    # THIS IS WRONG !
    graph.es["weight"] = [np.exp(-1*weight/args.sigma) for row in dists for weight in row]

if not graph.is_connected():
    print '\033[31mGraph is disconnected \033[00m'
else:
    print '\033[92mGraph is connected \033[00m'


print 'Saving graph...'
graph.save(args.prefix + '-graph.pickle', format="pickle")

print 'Saving kd-tree...'
flann.save_index(args.prefix + '-index.knn')
