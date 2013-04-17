import random
import operator
import numpy as np
from operator import *
from itertools import *
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.agent.Agent import Agent
from rlglue.utils import TaskSpecVRLGLUE3
from pyrl.basis.fourier import FourierBasis
from rlglue.environment.Environment import Environment

class Option:
    """ A Markov option o consists of a tuple:
    :math:`langle \mathcal{I}, \pi, \beta \rangle`

    where the initiation set :math:`\mathcal{I}` specifies in which states the
    option can be invoked, :math:`\pi` is a sub-policy defined for
    o and :math:`\beta` defines the probability of terminating
    the execution of :math:`\pi` in any given state.

    The options framework was originally introduced in:

    R. S. Sutton, D. Precup, and S. Singh, "Between MDPs and semi-MDPs:
    A framework for temporal abstraction in reinforcement learning,"
    Artificial Intelligence, vol. 112, no. 1-2, pp. 181-211, 1999.

    """

    def initiate(self, observation):
        """ Initiation predicate

        :param observation: the current observation
        :returns: True if this option can be taken in the current observation
        :rtype: bool

        """

    def terminate(self, observation):
        """ Termination (beta) function

        :param observation: the current observation
        :returns: True if this option must terminate in the current observation
        :rtype: bool or float defined as a proper probability function

        """

    def pi(self, observation):
        """ A deterministic greedy policy with respect to the approximate
        action-value function. Please note that we generally don't want to
        learn a policy over non-stationary options. Exploration strategies over
        primitive actions are therefore not needed in this case.

        :param observation: the current observation
        :returns: the greedy action with respect to the action-value function of
        this option
        :rtype: int

        """
class PrimitiveOption(Option):
    """ This class wraps a primitive discrete action into an option.

    A primitive action can be seen as a special kind of option whose
    policy always selects it whenever it is available, can be initiated
    everwhere and only last one step.

    """

    def __init__(self, action):
        self.action = action

    def initiate(self, observation):
        return True

    def terminate(self, observation):
        return True

    def pi(self, observation):
        return self.action

class KNNOption(Option):
    """ K-Nearest Neighbor Option

    Here we implement an option construction based on the community structures
    of a graph over continuous observations. A k-nearest neighbor index is used
    to control the initiation and termination conditions.

    More details can be found in:

    P. Bacon and D. Precup. "Using label propagation for learning
    temporally abstract actions in reinforcement learning." In Proceedings of
    the Workshop on Multiagent Interaction Networks (MAIN 2013), 2013.

    """
    def __init__(self, source, target):
        """ Create an option to navigate from a community to another

        :param source: The source community
        :type source: int
        :param target: The target community
        :type target: int

        """
        self.source_community = source
        self.target_community = target
        self.membership = None
        self.weights = None
        self.index = None
        self.basis = None

    def initiate(self, observation):
        """ Initiation predicate

        An option can be initiated if it contains a observation which is the closest the current observation.

        :param observation: the current observation
        :returns: True if this option can be taken in the current observation
        :rtype: bool

        """
        knn_idx, dist = self.index.nn_index(observation)
        return self.membership[knn_idx] == self.source_community

    def terminate(self, observation):
        """ Termination (beta) function

        The definition of beta that we adopt here makes it either 0 or 1.
        It returns 1 whenever its closest neighbor belongs to a different community.

        :param observation: the current observation
        :returns: True if this option must terminate in the current observation
        :rtype: bool

        """
        knn_idx, dist = self.index.nn_index(observation)
        return self.membership[knn_idx] != self.source_community

    def pi(self, observation):
        """ A deterministic greedy policy with respect to the approximate
        action-value function. Please note that we generally don't want to
        learn a policy over non-stationary options. Exploration strategies over
        primitive actions are therefore not needed in this case.

        :param observation: the current observation
        :returns: the greedy action with respect to the action-value function of
        this option
        :rtype: int

        """
        return np.dot(self.weights.T, np.array(self.basis.computeFeatures(observation))).argmax()

    def __getstate__(self):
        """ Implement pickling manually to avoid duplicating dataset

        """
        odict = self.__dict__.copy()
        del odict['membership']
        del odict['index']
        return odict

    def __setstate__(self, dict):
        """ Load the pickled object state

        The membership vector and index will have to be re-loaded manually.

        """
        self.__dict__.update(dict)
        self.membership = None
        self.index = None

class IntraOptionLearning(Agent):
    """ This class implements Intra-Option learning with
    linear function approximation.

    R. S. Sutton, D. Precup, and S. Singh, "Intra-option learning about
    temporally abstract actions," In Proceedings of the Fifteenth
    International Conference on Machine Learning (ICML 1998), 1998, pp. 556-564.

    """

    def __init__(self, options):
        """
        :param options: A set of options with learnt policies
        :type options: list of Option

        """
        self.alpha = 0.1
        self.gamma = 0.1
        self.epsilon = 0.1
        self.fa_order = 4
        self.options = options
        self.current_option = None
        self.finished_learning = False

    def intraoption_update(self, reward, features, observation):
        """ Perform a step of intra-option learning

        :param reward: The reward just obtained
        :param features: The features representation of the current state

        """
        for i in self.consistent_options():
            # Given that the current option must terminate, find which
            # possible next option has the highest value
            current_value = 0.0
            if self.options[i].terminate(observation):
                initializable_options = self.initializable_options(observation)

                current_value = max(izip(np.dot(self.weights[initializable_options], features),
                        initializable_options), key=itemgetter(0))[1]
            else:
                current_value = np.dot(self.weights[i], features)

            delta = reward + self.gamma*current_value - np.dot(self.weights[i], self.last_features)

            self.weights[i] = self.weights[i] + self.alpha*delta*self.last_features

    def consistent_options(self):
        """
        :returns: a subset of the options for which pi_o(s) = a
        :rtype: list of int
        """
        return filter(lambda idx: self.options[idx].pi(self.last_observation) == self.last_action,
                        xrange(len(self.options)))


    def initializable_options(self, observation):
        """ Find the options available under the current state

        :retuns: The indices of the options that can be initiated under the current state
        :rtype: list of int

        """
        return filter(lambda idx: self.options[idx].initiate(observation), xrange(len(self.options)))

    def egreedy(self, observation, features):
        """ Use epsilon-greedy exploration for the behavior policy

        :param observation: The raw observations
        :param features: The features representation of the observation
        :returns: A random option with probability epsilon, or the option with
        the highest value with probability 1 - epsilon.
        :rtype: int

        """
        initializable_options = self.initializable_options(observation)

        if random.random() < self.epsilon:
            return random.choice(initializable_options)

        return max(izip(np.dot(self.weights[initializable_options], features), initializable_options),
                        key=itemgetter(0))[1]

    def mu(self, observation, features=None):
        """ The semi-markov deterministic policy that follows
        an option to completion before starting another one.

        :param observation: The raw observations
        :param features: The features representation of the observation
        :returns: the best option according to the current policy
        :rtype: Option

        """
        if self.current_option == None or self.current_option.terminate(observation):
            self.current_option = self.options[self.egreedy(observation, features)]

        return self.current_option

    def agent_init(self, taskspec):
        """ This function is called once at the begining of an episode.
        Performs sanity checks with the environment.

        :param taskspec: The task specifications
        :type taskspec: str

        """
        spec = TaskSpecVRLGLUE3.TaskSpecParser(taskspec)
        if len(spec.getIntActions()) != 1:
            raise Exception("Expecting 1-dimensional discrete actions")
        if len(spec.getDoubleActions()) != 0:
            raise Exception("Expecting no continuous actions")
        if spec.isSpecial(spec.getIntActions()[0][0]):
            raise Exception("Expecting min action to be a number not a special value")
        if spec.isSpecial(spec.getIntActions()[0][1]):
            raise Exception("Expecting max action to be a number not a special value")

        observation_ranges = spec.getDoubleObservations()
        self.basis = FourierBasis(len(observation_ranges), self.fa_order, observation_ranges)
        self.weights = np.zeros((len(self.options), self.basis.numTerms))

        self.last_action = 0
        self.last_features = []
        self.last_observation = []

    def agent_start(self, obs):
        """ This function is called by the environment in the initial state.

        :param obs: An observation from the environment
        :rtype obs: :class:`rlglue.types.Observation`
        :returns: The primitive action to execute in the environment according to the
        behavior policy.
        :rtype: a primitive action under the form of a :class:`rlglue.types.Action`

        """
        observation = np.array(list(obs.doubleArray))
        current_features = self.basis.computeFeatures(observation)

        self.last_observation = observation
        self.last_features = current_features
        self.last_action = self.mu(observation, current_features).pi(observation)

        action = Action()
        action.intArray = [self.last_action]
        return action

    def agent_step(self, reward, obs):
        """ This function is called by the environment while the episode lasts.

        If learning is not frozen, the option-value function Q(s, o) is updated
        using intra-option learning.

        :param reward: The reward obtained as a result of the last transition.
        :param obs: An observation from the environment
        :rtype obs: :class:`rlglue.types.Observation`
        :returns: The primitive action to execute in the environment according to the
        behavior policy.
        :rtype: a primitive action under the form of a :class:`rlglue.types.Action`

        """
        observation = np.array(list(obs.doubleArray))
        current_features = self.basis.computeFeatures(observation)

        if not self.finished_learning:
            self.intraoption_update(reward, current_features, observation)

        self.last_observation = observation
        self.last_features = current_features
        self.last_action = self.mu(observation, current_features).pi(observation)

        action = Action()
        action.intArray = [self.last_action]
        return action

    def agent_end(self, reward):
        """ This function is called by the environment when the episode finishes.

        If learning is not frozen, the option-value function Q(s, o) is updated
        using intra-option learning.

        :param reward: The reward obtained as a result of the last transition.

        """
        if not self.finished_learning:
            for i in self.consistent_options():
                delta = reward - np.dot(self.weights[i], self.last_features)
                self.weights[i] = self.weights[i] + self.alpha*delta*self.last_features

    def agent_cleanup(self):
        pass

    def agent_message(self, msg):
        return "Intra-Option Learning does not understand your message."

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
        knn_idx, dist = self.kdtree.nn_index(np.array([observation]))
        return self.membership[knn_idx] == self.target_community

    def env_step(self, action):
        returnRO = self.decorated.env_step(action)

        # Set pseudo-termination
        if self.episode_ended(returnRO.o.doubleArray):
            returnRO.r = self.END_EPISODE
            returnRO.terminal = True

        return returnRO

    def env_init(self):
        return self.decorated.env_init()

    def env_start(self):
        return self.decorated.env_start()

    def env_cleanup(self):
        self.decorated.env_cleanup()

    def env_message(self, message):
        return self.decorated.env_message(message)
