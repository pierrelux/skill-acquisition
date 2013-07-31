#!/usr/bin/python
import os
import pp
import time
import options
import cPickle
import smtplib
import argparse

def learn_policy(options_learned, environment_name, nepisodes, max_steps, prefix):
    import csv
    import cPickle
    import options
    from pyrl.rlglue import RLGlueLocal as RLGlueLocal
    from pyrl.environments.pinball import PinballRLGlue

    agent = options.IntraOptionLearning(options_learned, alpha=0.01, gamma=1.0, epsilon=0.1, fa_order=4)
    environment = options.TrajectoryRecorder(PinballRLGlue(environment_name), prefix + '-trajectory')

    # Connect to RL-Glue
    rlglue = RLGlueLocal.LocalGlue(environment, agent)
    rlglue.RL_init()

    scores = []
    for i in xrange(nepisodes):
        print 'Episode ', i
        terminated = rlglue.RL_episode(max_steps)
        total_steps = rlglue.RL_num_steps()
        total_reward = rlglue.RL_return()

        print '\t %d steps, %d reward, %d terminated'%(total_steps, total_reward, terminated)
        score = [i, total_steps, total_reward, terminated]
        scores.append(score)

        with open(prefix + '-behavior-policy.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(score)
            f.flush()

    rlglue.RL_cleanup()

    cPickle.dump(agent, open(prefix + '-behavior-policy.pl', 'wb'))

    return scores

def job_done(msg):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('pierrelucbacon@gmail.com', 'password')
    server.sendmail('pierrelucbacon@gmail.com', 'phone@msg.telus.com', msg)

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
    parser.add_argument('--primitive', action='store_true', help='Add primitve actions')
    parser.add_argument('-a', '--number-agents', type=int, default=100, help='number of agents to average over')
    parser.add_argument('-p', '--prefix', action='store', type=str,
                    dest='prefix', help="output prefix (default: dataset)")
    args = parser.parse_args()

    if not args.prefix:
        args.prefix = os.path.splitext(os.path.basename(args.options))[0]

    # Import options
    print 'Loading options...'
    start_time = time.time()
    options_learned = cPickle.load(open(args.options, 'rb'))
    print 'Learning with %d options'%(len(options_learned),)

    # Throw in primitive actions
    if args.primitive:
        print 'Using primitive actions.'
        options_learned.extend((options.PrimitiveOption(a) for a in range(0, 5)))

    #job_server = pp.Server(ppservers=("*",))

    # Learn the behavior policy
    #jobs = [job_server.submit(learn_policy,
    #           (options_learned, args.environment, args.nepisodes,
    #               args.max_steps, args.prefix + '-agent%d'%(agent,)))
    #                   for agent in xrange(args.number_agents)]

    #job_server.wait()
    #job_server.print_stats()

    #scores = [job() for job in jobs]
    #cPickle.dump(scores, open(args.prefix + '-behavior-policy-aggregated.pl', 'wb'))
    learn_policy(options_learned, args.environment, args.nepisodes, args.max_steps, args.prefix)

    #job_done('Experiment has finished')
