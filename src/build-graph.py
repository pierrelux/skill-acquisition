#!/usr/bin/python

import sys
import random
import argparse
import numpy as np
import igraph as ig
from pyflann import *

# Parse arguments
parser = argparse.ArgumentParser(description='Build the state-space graph and find community structures.')

parser.add_argument('dataset')
parser.add_argument('-p', '--plot', action='store_true', help='Plot the graph with a force layout (default: false)')
parser.add_argument('-s', '--sigma', action='store', type=float, default=0.5, dest='sigma', help='Sigma value (default: 0.5)')
parser.add_argument('-k', '--knn', action='store', type=int, default=25, dest='knn', help='Number of nearest neighbors (default: 25)')
parser.add_argument('-r', '---random-subsampling', action='store', type=int, dest='nsamples', help='Perform random sumbsampling')

args = parser.parse_args()

# Load csv file
dataset = np.loadtxt(sys.argv[1])

# Random sumbsampling
if args.nsamples:
  dataset = dataset[random.sample(xrange(0, np.alen(dataset)), args.nsamples), :]

# Compute nearest neighbors
flann = FLANN()
knn, dists = flann.nn(dataset, dataset, args.knn+1)
knn = knn[:, 1:]
dists = dists[:, 1:]

# Create the state-space graph
graph = ig.Graph()
graph.add_vertices(np.alen(dataset))
graph.add_edges([(i, j) for i in xrange(np.alen(knn)) for j in knn[i,:]])
graph.es["weight"] = [np.exp(-1*weight/args.sigma) for row in dists for weight in row]

# Find communities by weighted LPA
communities = graph.community_label_propagation("weight")

print 'Is graph connected ? ', graph.is_connected()

# Plot with force layout
if args.plot:
  ig.plot(graph, "manifold.pdf", layout="fr")
