import sys
import random
import argparse
import itertools
import numpy as np
import igraph as ig
from pyflann import *


# Parse arguments
parser = argparse.ArgumentParser(description='Subsample the dataset')

parser.add_argument('dataset')
parser.add_argument('-n', '--nsamples', action='store', type=int, dest='nsamples', help='Number of samples')
parser.add_argument('-p', '--prefix', action='store', type=str, dest='prefix', help="output prefix (default: dataset)")
args = parser.parse_args()

if not args.prefix:
    args.prefix = os.path.splitext(os.path.basename(args.dataset))[0]

# Load csv file
dataset = np.loadtxt(sys.argv[1])

# Random sumbsampling
dataset = dataset[random.sample(xrange(0, np.alen(dataset)), args.nsamples), :]
np.savetxt(args.prefix + '-subsampled.dat', dataset)
