#!/usr/bin/python

import pickle
import argparse


class Option:

    def __init__(self, source, target, membership, kdtree):
        """ Create an option to naviage from a community to another
        @param source: the source community
        @param target: the target community
        """
        self.source_community = source
        self.target_community = target
        self.membership = membership
        self.kdtree = kdtree

    def can_initiate(self, state):
        """ Initiation predicate

        An option can be initiated if it contains a state which is the closest the current state.

        @param state: the current state

        @return True if this option can be taken in the current state
        """
        knn_idx, dist = self.kdtree.nn_index(state)
        return self.membership[knn_idx] == self.source_community

    def terminate(self, state):
        """ Termination (beta) function

        The definition of beta that we adopt here makes it either 0 or 1.
        It returns 1 whenever its closest neighbor belongs to a different community.

        @param state: the current state

        @return True if this option must terminate in the current state
        """
        knn_idx, dist = self.kdtree.nn_index(state)
        return self.membership[knn_idx] != self.source_community

parser = argparse.ArgumentParser(description='Create options from the community structures')
parser.add_argument('clustering', help='clustering obtained from the community detection algorithm')
args = parser.parse_args()

# Open the vertex dendogram
vd = pickle.load(open(args.clustering, 'rb'))

# Use the clustering with the highest modularity
cl = vd.as_clustering()

print cl.graph.ecount(), ' edges'
print cl.graph.vcount(), ' vertices'
print len(cl), ' communities'

# Find the neighboring communities
options_connectivity = set(((cl.membership[v1], cl.membership[v2]) for v1, v2 in cl.graph.get_edgelist()
                           if cl.membership[v1] != cl.membership[v2]))
options = [Option(source, target, cl.membership, None) for source, target in options_connectivity]
print len(options_connectivity), ' options'

# - Connect to RL-Glue
# - For each option, start at any of the initiation state
# - Run SARSA(Lambda) + FA until termination is triggered
