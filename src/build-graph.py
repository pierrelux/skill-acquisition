#!/usr/bin/python

import sys
import random
import argparse
import itertools
import numpy as np
import igraph as ig
from pyflann import *
from itertools import *


# Parse arguments
parser = argparse.ArgumentParser(description='Build the state-space graph')

parser.add_argument('dataset')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--directed', metavar='KNN', type=int, help="build a directed knn neighborhood graph")
group.add_argument('--mutual', metavar='KNN', type=int, help="build a mutual knn neighborhood graph")
group.add_argument('--symmetric', metavar='KNN', type=int, help="build a symmetric knn neighborhood graph")
group.add_argument('--radius', metavar='RADIUS', type=float, help="build a r-neighborhood graph")

group = parser.add_mutually_exclusive_group(required=False)
group.add_argument('--sigma', type=float, default=None, help='sigma value (default: 0.5)')
group.add_argument('--local-scaling', metavar='JTH', type=int, default=None, help="use local scaling according to Zelnik-Manor and Perona")

parser.add_argument('--delete', action='store_true', help="delete singleton vertices")
parser.add_argument('--prefix', action='store', type=str, dest='prefix', help="output prefix (default: dataset)")

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
    knn, dists = flann.nn_index(dataset, k+1)
    knn = knn[:, 1:]
    dists = dists[:, 1:]

    def heat_kernel(i, j):
        return np.exp(-1*dists[i,j]/args.sigma)

    def locally_scaled_heat_kernel(i, j, l):
        return np.exp(-1*dists[i,j]/(np.sqrt(dists[i,l])*np.sqrt(dists[knn[i,j],l])))

    if args.local_scaling:
        print 'Scaling locally'
        similarity = lambda i, j: locally_scaled_heat_kernel(i, j, args.local_scaling-1)
    else:
        similarity = heat_kernel

    if args.directed:
        print 'Building directed graph...'
        edges, weights = zip(*(((i, knn[i, j]), similarity(i, j)) for i in xrange(knn.shape[0])
            for j in xrange(knn.shape[1])))

    if args.mutual:
        print 'Building mutual graph...'
        edges, weights = zip(*(((i, knn[i, j]), similarity(i, j)) for i in xrange(knn.shape[0])
            for j in xrange(knn.shape[1]) if i in knn[knn[i, j],:]))

    if args.symmetric:
        print 'Building symmetric graph...'
        uniq_edges = dict(((frozenset((i, knn[i, j])), similarity(i, j)) for i in xrange(knn.shape[0])
            for j in xrange(knn.shape[1])))

        edges = map(list, uniq_edges.iterkeys())
        weights = uniq_edges.values()

    graph.add_edges(edges)
    graph.es['weight'] = weights

graph.vs['index'] = range(graph.vcount())

if graph.is_directed():
    components = graph.decompose(mode=ig.WEAK)
else:
    components = graph.decompose(mode=ig.WEAK)

if len(components) > 1:
    print '\033[31mGraph is disconnected\033[00m'

    if args.delete:
        spurious_vertices = []
        for component in islice(components, 1, None):
            spurious_vertices.extend(component.vs['index'])

        # Only keep the largest component
        graph = components[0]

        print '\033[92mGraph is now connected with %d vertices\033[00m'%(graph.vcount(),)

        # Update dataset and rebuild index
        flann = FLANN()
        dataset = np.delete(dataset, spurious_vertices, 0)

        print 'Reconstructing index...'
        flann.build_index(dataset)

        print 'Saving pruned dataset...'
        args.prefix = args.prefix + '-pruned'
        np.savetxt(args.prefix + '.dat', dataset)

else:
    print '\033[92mGraph is connected \033[00m'

print 'Saving graph...'
graph.save(args.prefix + '-graph.pickle', format="pickle")

print 'Saving kd-tree...'
flann.save_index(args.prefix + '-index.knn')
