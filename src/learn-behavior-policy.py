#!/usr/bin/python
import os
import csv
import cPickle
import argparse
from options import *
from pyflann import *
from pyrl.rlglue import RLGlueLocal as RLGlueLocal
from pyrl.environments.pinball import PinballRLGlue

def learn_policy(options, environment_name, nepisodes, max_steps, prefix):
    agent = IntraOptionLearning(options, alpha=0.01, gamma=1.0, epsilon=0.1, fa_order=4)
    environment = PinballRLGlue(environment_name)

    score_file = csv.writer(open(prefix + '-behavior-policy.csv', 'wb'))

    # Connect to RL-Glue
    rlglue = RLGlueLocal.LocalGlue(TrajectoryRecorder(environment, 'option-trajectory.dat'), agent)
    rlglue.RL_init()

    for i in xrange(nepisodes):
        print 'Episode ', i
        terminated = rlglue.RL_episode(max_steps)
        total_steps = rlglue.RL_num_steps()
        total_reward = rlglue.RL_return()

        print '\t %d steps, %d reward, %d terminated'%(total_steps, total_reward, terminated)
        score_file.writerow([i, total_steps, total_reward, terminated])

    rlglue.RL_cleanup()

    cPickle.dump(agent, open(prefix + '-behavior-policy.pl', 'wb'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learn the behavior policy over options')
    parser.add_argument('options', help='a set of options for this task')
    parser.add_argument('environment', help='environment configuration')
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
        args.prefix = os.path.splitext(os.path.basename(args.options))[0]

    # Import options
    print 'Loading options...'
    options = cPickle.load(open(args.options, 'rb'))
    print 'Learning with %d options'%(len(options),)

    # Learn the behavior policy
    learn_policy(options, args.environment, args.nepisodes, args.max_steps, args.prefix)
