#!/usr/bin/python
import os
import pp
import copy
import cPickle
import argparse
import itertools

def learn_option(source, target, dataset_filename, index_filename, clustering_filename, domain_filename, num_episodes=100, max_steps=10000, random_init=True):
    """
    :param source: the source community
    :type source: int
    :param target: the target community
    :param target: int
    """
    from pyrl.agents.sarsa_lambda import sarsa_lambda
    from pyrl.rlglue import RLGlueLocal as RLGlueLocal
    from pyrl.environments.pinball import PinballRLGlue
    import numpy as np
    import logging
    import pyflann
    import options
    import cPickle
    import random
    import csv

    prefix = 'option-%d-to-%d'%(source, target)

    logger = logging.getLogger(prefix)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(prefix + '.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Import dataset
    logger.info("Loading dataset")
    dataset = np.loadtxt(dataset_filename)

    # Import index
    logger.info("Loading index")
    flann = pyflann.FLANN()
    flann.load_index(index_filename, dataset)

    # Open the vertex dendogram
    logger.info("Loading clustering")
    vd = cPickle.load(open(clustering_filename, 'rb'))

    score_file = csv.writer(open(prefix + '-score.csv', 'wb'))
    # Use the clustering with the highest modularity
    cl = vd.as_clustering()
    print cl.graph.ecount(), ' edges'
    print cl.graph.vcount(), ' vertices'
    print len(cl), ' communities'

    # Create agent and environments
    agent = sarsa_lambda(epsilon=0.01, alpha=0.001, gamma=1.0, lmbda=0.9,
    params={'name':'fourier', 'order':4})

    environment = options.PseudoRewardEnvironment(PinballRLGlue(domain_filename), target, cl.membership, flann)

    # Connect to RL-Glue
    rlglue = RLGlueLocal.LocalGlue(environment, agent)
    rlglue.RL_init()

    # Execute episodes

    for i in xrange(num_episodes):
        if random_init:
            initial_state = dataset[random.choice(cl[source])]
            rlglue.RL_env_message('set-start-state %f %f %f %f'%(initial_state[0], initial_state[1], initial_state[2], initial_state[3]))
            logger.info("Set initial random state at %f %f %f %f"%(initial_state[0], initial_state[1], initial_state[2], initial_state[3]))

        logger.info("Starting episode %d of %d"%(i, num_episodes))
        terminated = rlglue.RL_episode(max_steps)

        total_steps = rlglue.RL_num_steps()
        total_reward = rlglue.RL_return()
        logger.info("%d steps, %d reward, %d terminated"%(total_steps, total_reward, terminated))

        with open(prefix + '-score.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([i, total_steps, total_reward, terminated])

    rlglue.RL_cleanup()
    logger.info("Learning terminated")

    option = options.KNNOption(source, target)
    option.weights = agent.weights[0,:,:]
    option.basis = agent.basis

    return option

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

    # Open the vertex dendogram
    print 'Loading clustering...'
    vd = cPickle.load(open(args.clustering, 'rb'))
    cl = vd.as_clustering()

    # Find the neighboring communities
    options_connectivity = [set(((cl.membership[v1], cl.membership[v2])
        for v1, v2 in cl.graph.get_edgelist() if cl.membership[v1] != cl.membership[v2]))]

    options_connectivity.extend([(target, source) for source, target in options_connectivity])

    print len(options_connectivity), ' options found'

    # Launch pp server with autodiscovery
    job_server = pp.Server(ppservers=("*",))
    job_server.set_ncpus(0)

    print 'Submitting jobs...'
    jobs = [job_server.submit(learn_option, (source, target,
            args.dataset, args.index, args.clustering, 'pinball_hard_single.cfg', args.nepisodes, args.max_steps))
            for source, target in options_connectivity]

    print 'Number of cpus: ', job_server.get_ncpus()
    job_server.print_stats()

    print 'Waiting for result...'
    job_server.wait()

    options = [job() for job in jobs]

    job_server.print_stats()

    # Throw in primitive actions
    options.extend((options.PrimitiveOption(a) for a in range(0, 5)))

    # Serialize options
    cPickle.dump(options, open(args.prefix + '-options.pl', 'wb'))
