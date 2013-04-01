#!/usr/bin/python
import os
import sys
import csv
import copy
import cPickle
import random
import argparse
import multiprocessing
import multiprocessing.managers

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

    def __init__(self, decorated, target, manager):
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
	self.manager = manager

    def episode_ended(self, observation):
        """ Check if the agent has reached the target region

	:returns: True if the nearest neighbor is in the target community
	:rtype: bool

	"""
        knn_idx, dist = self.manager.nn_index(np.array([observation]))._getvalue()
        return self.manager.membership(knn_idx)._getvalue() == self.target_community

    def env_step(self, action):
	returnRO = self.decorated.env_step(action)
	returnRO.terminal = self.episode_ended(returnRO.o.doubleArray)
	if returnRO.terminal:
	    returnRO.r = self.END_EPISODE

	return returnRO

    def env_init(self):
	return self.decorated.env_init()

    def env_start(self):
	return self.decorated.env_start()

    def env_cleanup(self):
	self.decorated.env_cleanup()

    def env_message(self, message):
	return self.decorated.env_message(message)

def learn_option(option):
    print 'Learning option from %d to %d'%(option.source_community, option.target_community)
    sys.stdout.flush()

    class IndexManager(multiprocessing.managers.BaseManager): pass
    IndexManager.register('nn_index')
    IndexManager.register('membership')
    IndexManager.register('random_node')
    IndexManager.register('nepisodes')
    IndexManager.register('max_steps')
    IndexManager.register('prefix')
    manager = IndexManager(address=('', 8081), authkey='dist')
    manager.connect()

    nepisodes = int(manager.nepisodes()._getvalue())
    max_steps = int(manager.max_steps()._getvalue())
    prefix = str(manager.prefix()._getvalue())

    agent = sarsa_lambda(epsilon=0.01, alpha=0.001, gamma=1.0, lmbda=0.9,
    params={'name':'fourier', 'order':4})

    environment = PseudoRewardEnvironment(PinballRLGlue('pinball_hard_single.cfg'),
    option.target_community, manager)

    # Connect to RL-Glue
    print 'Creating a local RL-Glue binding...'
    sys.stdout.flush()

    rlglue = RLGlueLocal.LocalGlue(environment, agent)
    rlglue.RL_init()

    # Execute episodes
    print 'Opening scores files...'
    sys.stdout.flush()

    score_file = csv.writer(open(prefix + '-score.csv', 'wb'))

    print 'Scores files opened'
    sys.stdout.flush()

    for i in xrange(nepisodes):
        # Use all of the nodes in the source community as possible
        # initial positions. Selected at random in each episode
        #start_position = dataset[random.choice(community)][:2]
	start_position = manager.random_node(option.source_community)._getvalue()[:2]
	print 'Got random position at ', start_position
        sys.stdout.flush()

        rlglue.RL_env_message('set-start-state %f %f'%(start_position[0], start_position[1]))
        print '\tEpsiode %d of %d starting at %f, %f'%(i, nepisodes, start_position[0], start_position[1])
        sys.stdout.flush()

        terminated = rlglue.RL_episode(max_steps)

        total_steps = rlglue.RL_num_steps()
        total_reward = rlglue.RL_return()
        print '\t\t %d steps, %d reward, %d terminated'%(total_steps, total_reward, terminated)
        sys.stdout.flush()

        score_file.writerow([i, total_steps, total_reward, terminated])

    option.weights = agent.weights.copy()
    option.basis = copy.deepcopy(agent.basis)
    rlglue.RL_cleanup()

    #cPickle.dump(options, open(prefix + '-options.pl', 'wb'))
    print "Returning option"
    sys.stdout.flush()

    return option

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

    # Expose the kd-tree over network
    class IndexManager(multiprocessing.managers.BaseManager): pass
    IndexManager.register('nn_index', callable=lambda pts: index.nn_index(pts))
    IndexManager.register('membership', callable=lambda n: cl.membership[n])
    IndexManager.register('random_node', callable=lambda c: dataset[random.choice(cl[c])])
    IndexManager.register('nepisodes', callable=lambda: num_episodes)
    IndexManager.register('max_steps', callable=lambda: max_steps)
    IndexManager.register('prefix', callable=lambda: prefix)

    manager = IndexManager(address=('', 8081), authkey='dist')
    print 'Starting manager...'
    manager.start()

    print 'Creating process pool...'
    pool = multiprocessing.Pool()
    print 'Parallel map...'
    options = pool.map(learn_option, options)
    print 'Joining...'
    pool.join()
    print 'Shudown...'
    manager.shutdown()

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

