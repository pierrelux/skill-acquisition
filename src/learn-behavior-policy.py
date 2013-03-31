#!/usr/bin/python

import os
import csv
import cPickle
import argparse
from options import *
from pyflann import *
from pyrl.rlglue import RLGlueLocal as RLGlueLocal
from pyrl.environments.pinball import PinballRLGlue

def learn_policy(options, index, nepisodes, max_steps):
    agent = IntraOptionLearning(options)
    environment = PinballRLGlue('pinball_hard_single.cfg')

    # Connect to RL-Glue
    rlglue = RLGlueLocal.LocalGlue(environment, agent)
    rlglue.RL_init()

    for i in xrange(nepisodes):
	terminated = rlglue.RL_episode(max_steps)
	total_steps = rlglue.RL_num_steps()
	total_reward = rlglue.RL_return()

	print '\t\t %d steps, %d reward, %d terminated'%(total_steps, total_reward, terminated)
	score_file.writerow([i, total_steps, total_reward, terminated])

	rlglue.RL_cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learn the behavior policy over options')
    parser.add_argument('dataset', help='input dataset used to build the index')
    parser.add_argument('index', help='an k-nearest neighbor index')
    parser.add_argument('clustering', help='clustering obtained from the\
		    community detection algorithm')
    parser.add_argument('options', help='a set of options for this task')
    parser.add_argument('-n', '--number-episodes', dest='nepisodes', type=int,
		    default=100, help='the number of episodes to execute for\
		    learning the policy over options (default: 100)')
    parser.add_argument('-s', '--max-steps', dest='max_steps', type=int,
		    default=10000, help='the maximum number of steps that the\
		    agent is allowed to take in the environment')
    parser.add_argument('-p', '--prefix', action='store', type=str,
		    dest='prefix', help="output prefix (default: dataset)")
    args = parser.parse_args()

    if not args.prefix:
        args.prefix = os.path.splitext(os.path.basename(args.dataset))[0]

    # Import dataset
    print 'Loading dataset...'
    dataset = np.loadtxt(args.dataset)

    # Import index
    print 'Loading index...'
    flann = FLANN()
    flann.load_index(args.index, dataset)

    # Open the vertex dendogram
    print 'Loading clustering...'
    vd = cPickle.load(open(args.clustering, 'rb'))
    cl = vd.as_clustering()

    # Import options
    print 'Loading options...'
    options = cPickle.load(open(args.options, 'rb'))
    for option in options:
	option.index = flann
	option.membership = cl.membership

    # Learn the behavior policy
    learn_policy(options, args.nepisodes, args.max_steps)

