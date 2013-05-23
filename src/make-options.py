import os
import options
import cPickle
import argparse
import numpy as np
from pyflann import *
from sklearn import linear_model

def make_knnoptions(args):
    if not args.prefix:
        args.prefix = os.path.splitext(os.path.basename(args.dataset))[0]

    vd = cPickle.load(open(args.clustering, 'rb'))
    cl = vd.as_clustering()

    cPickle.dump([options.KNNOption(label=cl.membership[cl_graph.vs['index'][0]],
        membership=np.array(cl.membership), index_filename=args.index,
        dataset_filename=args.dataset, nn=args.nn) for cl_graph in cl.subgraphs()],
        open(args.prefix + '-options.pl', 'wb'))

def make_logisticoptions(args):
    if not args.prefix:
        args.prefix = os.path.splitext(os.path.basename(args.dataset))[0]

    vd = cPickle.load(open(args.clustering, 'rb'))
    cl = vd.as_clustering()
    dataset = np.loadtxt(args.dataset)

    logreg = linear_model.LogisticRegression()
    logreg.fit(dataset, cl.membership)

    cPickle.dump([options.LogisticOption(label=cl.membership[cl_graph.vs['index'][0]],
        predictor=logreg, initial_states=dataset[cl_graph.vs['index']]) for cl_graph in cl.subgraphs()],
        open(args.prefix + '-options.pl', 'wb'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create options from the structures in the state-space')

    subparsers = parser.add_subparsers(title='option construction type')
    knn_parser = subparsers.add_parser('knn')
    knn_parser.add_argument('--nn', type=int, default=1, help="number of nearest neighbors (default: 1)")
    knn_parser.add_argument('dataset', help='input dataset used to build the index')
    knn_parser.add_argument('clustering', help='dataset clustering')
    knn_parser.add_argument('index', help='nearest neighbor index')
    knn_parser.set_defaults(func=make_knnoptions)

    logistic_parser = subparsers.add_parser('logistic')
    logistic_parser.add_argument('--regularizer', type=float, default=1.0, help="regularizer (default: 1.0)")
    logistic_parser.add_argument('dataset', help='input dataset used to build the index')
    logistic_parser.add_argument('clustering', help='input dataset used to build the index')
    logistic_parser.set_defaults(func=make_logisticoptions)

    parser.add_argument('--prefix', action='store', type=str, dest='prefix', help="output prefix (default: dataset)")
    args = parser.parse_args()

    args.func(args)


