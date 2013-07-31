#!/usr/bin/python
import os
import pp
import csv
import copy
import cPickle
import options
import argparse
import itertools
import numpy as np

def learn_option(option, environment_name, num_episodes, max_steps):
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

    prefix = 'option-%d-to-%d'%(option.label, option.target)
    score_file = csv.writer(open(prefix + '-score.csv', 'wb'))

    # Create agent and environments
    agent = sarsa_lambda(epsilon=0.01, alpha=0.001, gamma=0.9, lmbda=0.9,
    params={'name':'fourier', 'order':4})

    # Wrap the environment with the option's pseudo-reward
    environment = options.TrajectoryRecorder(options.PseudoRewardEnvironment(PinballRLGlue(environment_name), option, 10000), prefix + '-trajectory')

    # Connect to RL-Glue
    rlglue = RLGlueLocal.LocalGlue(environment, agent)
    rlglue.RL_init()

    # Execute episodes
    if not num_episodes:
        num_episodes = np.alen(option.initial_states)
        print 'Learning %d episodes'%(num_episodes,)

    for i in xrange(num_episodes):
        initial_state = option.initial_state()
        rlglue.RL_env_message('set-start-state %f %f %f %f'
               %(initial_state[0], initial_state[1], initial_state[2], initial_state[3]))

        terminated = rlglue.RL_episode(max_steps)

        total_steps = rlglue.RL_num_steps()
        total_reward = rlglue.RL_return()

        with open(prefix + '-score.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([i, total_steps, total_reward, terminated])

    rlglue.RL_cleanup()

    # Save function approximation
    option.basis = agent.basis
    option.weights = agent.weights[0,:,:]

    cPickle.dump(option, open(prefix + '-policy.pl', 'wb'))

    return option

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learn options in parallel')
    parser.add_argument('options', help='pickled options')
    parser.add_argument('environment', help='environment configuration')
    parser.add_argument('-n', '--number-episodes', dest='nepisodes', type=int,
		    default=None, help='the number of episodes to execute for\
		    learning each option (default: auto)')
    parser.add_argument('-s', '--max-steps', dest='max_steps', type=int,
		    default=10000, help='the maximum number of steps that the\
		    agent is allowed to take in the environment')
    parser.add_argument('-p', '--prefix', action='store', type=str,
		    dest='prefix', help="output prefix (default: dataset)")
    args = parser.parse_args()
    if not args.prefix:
        args.prefix = os.path.splitext(os.path.basename(args.options))[0]

    # Launch pp server with autodiscovery
    job_server = pp.Server(ppservers=("*",), ncpus=0, socket_timeout=None)

    print 'Submitting jobs...'
    jobs = [job_server.submit(learn_option, (option, args.environment, args.nepisodes, args.max_steps))
            for option in cPickle.load(open(args.options, 'rb'))]

    print 'Number of cpus: ', job_server.get_ncpus()
    print 'Waiting for result...'
    job_server.wait()
    job_server.print_stats()

    subgoal_options = [job() for job in jobs]

    # Serialize options
    cPickle.dump(subgoal_options, open(args.prefix + '-learned.pl', 'wb'))
