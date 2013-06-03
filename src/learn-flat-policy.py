#!/usr/bin/python
import os
import pp
import cPickle
import argparse

def learn_policy(environment_name, nepisodes, max_steps, prefix):
    from pyrl.agents.sarsa_lambda import sarsa_lambda
    from pyrl.rlglue import RLGlueLocal as RLGlueLocal
    from pyrl.environments.pinball import PinballRLGlue
    from options import TrajectoryRecorder
    import cPickle
    import csv

    # Create agent and environments
    agent = sarsa_lambda(epsilon=0.01, alpha=0.001, gamma=1.0, lmbda=0.9,
    params={'name':'fourier', 'order':4})

    # Wrap the environment with the option's pseudo-reward
    environment = TrajectoryRecorder(PinballRLGlue(environment_name), prefix + '-trajectory')

    score_file = csv.writer(open(prefix + '-scores.csv', 'wb'))

    # Connect to RL-Glue
    rlglue = RLGlueLocal.LocalGlue(environment, agent)
    rlglue.RL_init()

    # Execute episodes
    scores = []
    for i in xrange(nepisodes):
        print 'Episode ', i
        terminated = rlglue.RL_episode(max_steps)
        total_steps = rlglue.RL_num_steps()
        total_reward = rlglue.RL_return()

        print '\t %d steps, %d reward, %d terminated'%(total_steps, total_reward, terminated)
        score = [i, total_steps, total_reward, terminated]
        scores.append(score)
        score_file.writerow(score)

    rlglue.RL_cleanup()

    cPickle.dump(agent, open(prefix + '.pl', 'wb'))

    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learn the behavior policy over options')
    parser.add_argument('environment', help='environment configuration')
    parser.add_argument('-n', '--number-episodes', dest='nepisodes', type=int,
                    default=100, help='the number of episodes to execute for\
                    learning the policy over options (default: 100)')
    parser.add_argument('-s', '--max-steps', dest='max_steps', type=int,
                    default=10000, help='the maximum number of steps that the\
                    agent is allowed to take in the environment')
    parser.add_argument('-a', '--number-agents', type=int, default=100, help='number of agents to average over')
    parser.add_argument('-p', '--prefix', action='store', type=str,
                    dest='prefix', help="output prefix (default: dataset)")
    args = parser.parse_args()

    if not args.prefix:
        args.prefix = os.path.splitext(os.path.basename(args.environment))[0]

    # Launch pp server with autodiscovery
    job_server = pp.Server(ppservers=("*",))

    print 'Submitting jobs...'
    jobs = [job_server.submit(learn_policy, (args.environment, args.nepisodes, args.max_steps, args.prefix + '-flat-policy-%d'%(agent,))) for agent in xrange(args.number_agents)]

    print 'Number of cpus: ', job_server.get_ncpus()
    print 'Waiting for result...'
    job_server.wait()
    job_server.print_stats()

    scores = [job() for job in jobs]
    cPickle.dump(scores, open(args.prefix + '-aggregated.pl', 'wb'))

