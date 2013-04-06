#!/usr/bin/python

import os
import csv
import copy
import random
import cPickle
import argparse
import itertools

from options import *
from pyflann import *
from pyrl.agents.sarsa_lambda import sarsa_lambda
from pyrl.rlglue import RLGlueLocal as RLGlueLocal
from pyrl.environments.pinball import PinballRLGlue
from rlglue.environment.Environment import Environment

class PseudoRewardEnvironment(Environment):
    """ This class is a decorator for an RL-Glue environment.

    We need it to change at run time the terminal states of the
    underlying environment, and imposing our pseudo-reward function
    for the sake of learning a give policy for an option.

    """
    END_EPISODE = 10000

    def __init__(self, decorated, target, membership, kdtree):
        """
        :param decorated: The base environment
        :type decorated: Environment
        :param target: The target community to reach
        :type target: int
        :param membership: A vector of the community membership for each node
        :type membership: list
        :param kdtree: A KD-Tree index
        :type kdtree: FLANN

        """
        self.decorated = decorated
        self.target_community = target
        self.membership = membership
        self.kdtree = kdtree

    def episode_ended(self, observation):
        """ Check if the agent has reached the target region

        :returns: True if the nearest neighbor is in the target community
        :rtype: bool

        """
        knn_idx, dist = self.kdtree.nn_index(np.array([observation]))
        return self.membership[knn_idx] == self.target_community

    def env_step(self, action):
        returnRO = self.decorated.env_step(action)

        # Set pseudo-termination
        if self.episode_ended(returnRO.o.doubleArray):
            returnRO.r = self.END_EPISODE
            returnRO.terminal = True

        return returnRO

    def env_init(self):
	return self.decorated.env_init()

    def env_start(self):
	return self.decorated.env_start()

    def env_cleanup(self):
	self.decorated.env_cleanup()

    def env_message(self, message):
	return self.decorated.env_message(message)

def learn_options(dataset, index, cl, num_episodes, max_steps, prefix):
    """ From the graph clustering step, learn options to navigate between adjacent communities.

    :param dataset: The points used build the graph
    :type dataset: numpy.array
    :param index: A KD-Tree index
    :type index: FLANN
    :param cl: The community clustering
    :type cl: igraph.VertexClustering
    :param num_episodes: The number of training episodes
    :type num_episodes: int
    :param max_steps: The maximum number of steps that the agent is allowed
    to take in the environment
    :type max_steps: int

    """

    # Find the neighboring communities
    options_connectivity = set(((cl.membership[v1], cl.membership[v2]) for v1, v2 in cl.graph.get_edgelist() if cl.membership[v1] != cl.membership[v2]))

    # Make options
    options = [Option(source, target) for source, target in options_connectivity]
    print len(options_connectivity), ' options found'

    # Learn options
    for option, idx in itertools.izip(options, count()):
        print 'Learning option %d of %d from %d to %d'%(idx, len(options), option.source_community, option.target_community)

        agent = sarsa_lambda(epsilon=0.01, alpha=0.001, gamma=1.0, lmbda=0.9,
        params={'name':'fourier', 'order':4})

        environment = PseudoRewardEnvironment(PinballRLGlue('pinball_hard_single.cfg'),
        option.target_community, cl.membership, index)

        # Connect to RL-Glue
        rlglue = RLGlueLocal.LocalGlue(environment, agent)
        rlglue.RL_init()

        # Execute episodes
        score_file = csv.writer(open(prefix + '-score' + str(idx) + '.csv', 'wb'))

        start_position = dataset[random.choice(cl[option.source_community])][:2]
        random_init = False

        for i in xrange(num_episodes):
            # Use all of the nodes in the source community as possible
            # initial positions. Selected at random in each episode
            if random_init:
                start_position = dataset[random.choice(cl[option.source_community])][:2]

            rlglue.RL_env_message('set-start-state %f %f'%(start_position[0], start_position[1]))

            print '\tEpsiode %d of %d starting at %f, %f'%(i, num_episodes, start_position[0], start_position[1])
            terminated = rlglue.RL_episode(max_steps)

            total_steps = rlglue.RL_num_steps()
            total_reward = rlglue.RL_return()
            print '\t\t %d steps, %d reward, %d terminated'%(total_steps, total_reward, terminated)

            score_file.writerow([i, total_steps, total_reward, terminated])

        option.weights = agent.weights.copy()
        option.basis = copy.deepcopy(agent.basis)
        rlglue.RL_cleanup()
        cPickle.dump(option, open(prefix + '-option' + str(idx) + '.pl', 'wb'))

    # Serialize options
    cPickle.dump(options, open(prefix + '-options.pl', 'wb'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create options from the\
		    community structures')
    parser.add_argument('dataset', help='Input dataset used to build the index')
    parser.add_argument('index', help='KNN index')
    parser.add_argument('clustering', help='clustering obtained from the\
		    community detection algorithm')
    parser.add_argument('-n', '--number-episodes', dest='nepisodes', type=int,
		    default=100, help='the number of episodes to execute for\
		    learning each option (default: 100)')
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

    # Use the clustering with the highest modularity
    cl = vd.as_clustering()
    print cl.graph.ecount(), ' edges'
    print cl.graph.vcount(), ' vertices'
    print len(cl), ' communities'

    # Create and learn options
    learn_options(dataset, flann, cl, args.nepisodes, args.max_steps, args.prefix)
