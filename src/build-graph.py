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
parser.add_argument('-k', '--knn', action='store', type=int, default=25, dest='knn', help='number of nearest neighbors (default: 25)')
parser.add_argument('-r', '---random-subsampling', action='store', type=int, dest='nsamples', help='perform random sumbsampling')
parser.add_argument('-o', '--output', action='store', type=str, default='graph', dest='output', help="output prefix (default: 'graph')")
args = parser.parse_args()

# Load csv file
dataset = np.loadtxt(sys.argv[1])

# Random sumbsampling
if args.nsamples:
  print 'Subsampling...'
  dataset = dataset[random.sample(xrange(0, np.alen(dataset)), args.nsamples), :]

# Compute nearest neighbors
print 'Building kd-tree index...'
flann = FLANN()
flann.build_index(dataset)

print 'Searching all nearest neighbors...'
knn, dists = flann.nn_index(dataset, args.knn+1)
knn = knn[:, 1:]
dists = dists[:, 1:]

# Create the state-space graph
print 'Building graph...'
graph = ig.Graph()
graph.add_vertices(np.alen(dataset))
graph.add_edges([(i, j) for i in xrange(np.alen(knn)) for j in knn[i,:]])
graph.es["weight"] = [np.exp(-1*weight/args.sigma) for row in dists for weight in row]

print 'Graph connected ? ', graph.is_connected()

print 'Saving graph...'
graph.save(args.output + '-graph.pickle', format="pickle")

print 'Saving kd-tree...'
flann.save_index(args.output + '-index.knn')
