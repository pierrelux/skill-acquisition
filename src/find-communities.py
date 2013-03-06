#!/usr/bin/python

import sys
import pickle
import argparse
import igraph as ig

parser = argparse.ArgumentParser(description='Find the community structures')
parser.add_argument('graph', help='the state-space graph')
parser.add_argument('-t', '--time', action='store', type=int, default=4, dest='time', help='length t of the random walks (default: 4)')
parser.add_argument('-o', '--output', action='store', type=str, default='communities', dest='output', help="output prefix (default: 'communities')")
args = parser.parse_args()

# Load graph
graph = ig.load(sys.argv[1])

# Compute community structure
vd = graph.community_walktrap(weights='weight', steps=args.time)

# Save clustering
pickle.dump(vd, open(args.output + '-cl.pickle', 'wb'))
