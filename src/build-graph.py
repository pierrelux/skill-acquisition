import sys
import numpy as np
import igraph as ig
from pyflann import *

# Load csv file
dataset = np.loadtxt(sys.argv[1])

# Compute nearest neighbors
k = 25
flann = FLANN()
knn, dists = flann.nn(dataset, dataset, k)
knn = knn[:, 1:]
dists = dists[:, 1:]

# Create the state-space graph
sigma = 0.5
graph = ig.Graph()
graph.add_vertices(np.alen(dataset))
graph.add_edges([(i, j) for i in xrange(np.alen(knn)) for j in knn[i,:]])
graph.es["weight"] = [np.exp(-1*weight/sigma) for row in dists for weight in row]

# Find communities by weighted LPA
communities = graph.community_label_propagation("weight")
print communities
print 'Is graph connected ? ', graph.is_connected()

# Plot with force layout
ig.plot(graph, "manifold.pdf", layout="fr")
