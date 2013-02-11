#!/bin/bash

if [[ $# < 2 ]]; then
 echo "$0 [MAXSTEPS] [EPISODES] "
 echo "Create dataset from qualifing trajectories"
 exit -1
fi

# Select terminated trajectories under a given number of steps
cat `awk '$1 < MAXSTEPS && $3==1 { print "pinball_observations" FNR ".dat" }' MAXSTEPS=$1 $2` | sort | uniq -u
