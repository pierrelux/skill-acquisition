#!/usr/bin/python

import os
import pickle
import argparse
import igraph as ig

parser = argparse.ArgumentParser(description='Find the community structures')
parser.add_argument('graph', help='the state-space graph')
parser.add_argument('-t', '--time', action='store', type=int, default=4, dest='time', help='length t of the random walks (default: 4)')
parser.add_argument('-p', '--prefix', action='store', type=str, dest='prefix', help="output prefix (default: graph)")
parser.add_argument('--weighted', action='store_true', help="Use weighted graph")
parser.add_argument('-v', '--verbose', action='store_true', help="Verbose mode")

args = parser.parse_args()
if not args.prefix:
    args.prefix = os.path.splitext(os.path.basename(args.graph))[0]

# Load graph
if args.verbose:
    print 'Loading graph...'

graph = ig.load(args.graph)

# Compute community structure
if args.verbose:
    print 'Finding communities...'

if args.weighted:
  vd = graph.community_walktrap(weights='weight', steps=args.time)
else:
  vd = graph.community_walktrap(steps=args.time)

cl = vd.as_clustering()
print '%d %d %f'%(args.time, len(cl), cl.modularity)

# Save clustering
if args.verbose:
    print 'Saving clustering...'

pickle.dump(cl.membership, open(args.prefix + '-membership.pickle', 'wb'))
pickle.dump(vd, open(args.prefix + '-clustering.pickle', 'wb'))
