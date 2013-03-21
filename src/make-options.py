#!/usr/bin/python

import copy
import time
import pickle
import argparse
import numpy as np

from pyflann import *
import rlglue.RLGlue as rl_glue
from pyrl.agents.sarsa_lambda import sarsa_lambda
from pyrl.rlglue import RLGlueLocal as RLGlueLocal
from pyrl.environments.pinball import PinballRLGlue
from rlglue.environment.Environment import Environment

class Option:

    def __init__(self, source, target, membership, kdtree):
        """ Create an option to navigate from a community to another

	:param source: The source community
	:type source: int
	:param target: The target community
	:type target: int
	:param membership: A vector of the community membership for each node
	:type membership: list
	:param kdtree: A KD-Tree index
	:type kdtree: FLANN
        """
        self.source_community = source
        self.target_community = target
        self.membership = membership
        self.kdtree = kdtree
	self.weights = None
	self.basis = None

    def can_initiate(self, state):
        """ Initiation predicate

        An option can be initiated if it contains a state which is the closest the current state.

	:param state: the current state
	:returns: True if this option can be taken in the current state
	:rtype: bool
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
        knn_idx, dist = self.kdtree.nn_index(observation)
        return self.membership[knn_idx] == self.target_community

    def env_step(self, action):
	returnRO = self.decorated.env_step(action)
	returnRO.terminal = self.episode_ended(returnRO.o.doubleArray)
	if returnRO.terminal:
	    returnRO.r = END_EPISODE

	return returnRO

    def env_init(self):
	return self.decorated.env_init()

    def env_start(self):
	return self.decorated.env_start()

    def env_cleanup(self):
	self.decorated.env_cleanup()

    def env_message(message):
	return self.decorated.env_message()

def timestamp():
    return time.strftime('%Y-%m-%d-%H-%M-%S.pl')

def learn_options(dataset, index, cl, num_episodes):
    """ From the graph clustering step, learn options to navigate between adjacent communities.

    :param dataset: The points used build the graph
    :type dataset: numpy.array
    :param index: A KD-Tree index
    :type index: FLANN
    :param cl: The community clustering
    :type cl: igraph.VertexClustering
    :param num_episodes: The number of training episodes
    :type num_episodes: int

    """
    MAX_STEPS = 10000

    # Find the neighboring communities
    options_connectivity = set(((cl.membership[v1], cl.membership[v2]) for v1, v2 in cl.graph.get_edgelist()
                               if cl.membership[v1] != cl.membership[v2]))
    # Make options
    options = [Option(source, target, cl.membership, None) for source, target in options_connectivity]
    print len(options_connectivity), ' options found'

    # Learn options
    for option in options:
	agent = sarsa_lambda(epsilon=0.01, alpha=0.001, gamma=1.0, lmbda=0.9, params={'name':'fourier', 'order':4})
	environment = PseudoRewardEnvironment(PinballRLGlue('pinball_simple_single.cfg'),
			option.target_community, cl.membership, index)

        # Connect to RL-Glue
        RLGlue = RLGlueLocal.LocalGlue(environment, agent)

        # Execute episodes
	option_file_prefix = 'option-from-%d-to-%d-%s'%(option.source_community, option.target_community, timestamp())
        score_file = csv.writer(open(option_file_prefix + '-score.csv', 'wb'))

	for i in xrange(num_episodes):
	    # Use all of the nodes in the source community as possible
            # initial positions. Selected at random in each episode
	    start_position = dataset[random.choice(cl[option.source])][:2]
            RLGlue.RL_env_message('set-start-state ' + start_position[0] + " " + start_position[1])

            terminated = RLGlue.RL_episode(MAX_STEPS)

	    total_steps = RLGlue.RL_num_steps()
            total_reward = RLGlue.RL_return()

            score_file.writerow([i, total_steps, total_reward, terminated])

        score_file.close()

	option.weights = agent.weights.copy()
	option.basis = copy.deepcopy(agent.basis)

    # Serialize options
    pickle(options, open(option_file_prefix + '.pl', 'wb'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create options from the community structures')
    parser.add_argument('dataset', help='Input dataset used to build the index')
    parser.add_argument('index', help='KNN index')
    parser.add_argument('clustering', help='clustering obtained from the community detection algorithm')
    parser.add_argument('-e', '--number-episodes', dest='nepisodes', type=int, default=100, help='the number of episodes to execute for learning each option (default: 100)')

    args = parser.parse_args()

    # Open the vertex dendogram
    print 'Loading clustering...'
    vd = pickle.load(open(args.clustering, 'rb'))

    # Use the clustering with the highest modularity
    cl = vd.as_clustering()
    print cl.graph.ecount(), ' edges'
    print cl.graph.vcount(), ' vertices'
    print len(cl), ' communities'

    # Import dataset
    print 'Loading dataset...'
    dataset = np.loadtxt(args.dataset)

    # Import index
    print 'Loading index...'
    flann = FLANN()
    flann.load_index(args.index, dataset)

    # Create and learn options
    #learn_options(dataset, flann, cl, args.nepisodes)
