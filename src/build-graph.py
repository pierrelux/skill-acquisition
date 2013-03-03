#!/usr/bin/python

import sys
import pickle
import random
import argparse
import itertools
import numpy as np
import igraph as ig
from pyflann import *
from scipy.cluster.vq import *


def lpa_aggregate(graph, nmerges=0):
  # Find communities by weighted LPA
  cl1 = graph.community_label_propagation("weight")
  labels1 = cl1.membership
  print 'Number of communities in solution 1: ', len(cl1)
  print 'Modularity score: ', cl1.modularity

  # Aggregate
  for i in xrange(nmerges):
    # Compute new clustering
    cl2 = graph.community_label_propagation("weight")
    labels2 = cl2.membership

    print 'Number of communities in solution %d: %d'%(i+2, len(cl2))
    print 'Modularity score: ', cl2.modularity

    # Combine labels
    unique_labels = list(set(itertools.izip(labels1, labels2)))
    relabling = dict(itertools.izip(unique_labels, xrange(len(unique_labels))))
    labels = [ relabling[l] for l in itertools.izip(labels1, labels2) ]

    # Compute new solution
    cl1 = graph.community_label_propagation(initial=labels);
    print 'Number of communities in aggregate solution: ', len(cl1)
    print 'Modularity score: ', cl1.modularity

  return cl1

def spectral_clustering(graph, k):
    # Compute the normalized Laplacian
    print 'Computing Laplacian...'
    laplacian = graph.laplacian(weights="weight", normalized=True)

    # Compute the Eigenvectors
    print 'Computing Eigenvectors...'
    w, v = np.linalg.eig(laplacian)

    # Apply K-Means
    print 'Applying K-Means...'
    kmeans(v[:, 0:k], k)

def compare(graph):
  print '\nLPA Method: '
  cl = lpa_aggregate(graph)

  print '\nInfomap method: '
  cl = graph.community_infomap(edge_weights="weight")
  print 'Number of communities: ', len(cl)
  print 'Modularity score: ', cl.modularity

  print '\nWalktrap method: '
  cl = graph.community_walktrap(weights="weight").as_clustering()
  print 'Number of communities: ', len(cl)
  print 'Modularity score: ', cl.modularity

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

# Compute communities
print 'Graph connected ? ', graph.is_connected()

#spectral_clustering(graph, 30)
compare(graph)
#graph.save("manifold.net", format="ncol")

# Plot with force layout
if args.plot:
   ig.plot(cl, "manifold.pdf", layout="large")
