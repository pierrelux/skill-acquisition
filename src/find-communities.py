#!/usr/bin/python

import os
import pickle
import argparse
import igraph as ig

parser = argparse.ArgumentParser(description='Find the community structures')
parser.add_argument('graph', help='the state-space graph')
parser.add_argument('-t', '--time', action='store', type=int, default=4, dest='time', help='length t of the random walks (default: 4)')
parser.add_argument('-p', '--prefix', action='store', type=str, dest='prefix', help="output prefix (default: graph)")
args = parser.parse_args()
if not args.prefix:
    args.prefix = os.path.splitext(os.path.basename(args.graph))[0]

# Load graph
print 'Loading graph...'
graph = ig.load(args.graph)

# Compute community structure
print 'Finding communities...'
vd = graph.community_walktrap(weights='weight', steps=args.time)

# Save clustering
print 'Saving clustering...'
pickle.dump(vd, open(args.prefix + '-clustering.pickle', 'wb'))
