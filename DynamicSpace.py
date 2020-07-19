import gym
from gym.spaces import MultiDiscrete
import numpy as np

#You could also inherit from Discrete or Box here and just override the shape(), sample() and contains() methods
class Dynamic(MultiDiscrete):
    """
    x where x in available actions {0,1,3,5,...,n-1}
    Example usage:
    self.action_space = spaces.Dynamic(max_space=2)
    """

    def __init__(self, nvec):
        super().__init__(nvec)
        #initially all actions are available
        self.available_actions = [{i for i in range(nvec[j])} for j in nvec]

    def disable_action(self, row, col):
        """ You would call this method inside your environment to remove available actions"""
        self.available_actions[row] = self.available_actions[row] - {col}
        return self.available_actions

    def enable_actions(self, row, col):
        """ You would call this method inside your environment to enable actions"""
        self.available_actions = self.available_actions[row] + {col}
        return self.available_actions

    def sample(self):
        row = np.randint(len(self.available_actions))
        return np.random.choice(self.available_actions[row])

    def contains(self, x):
        return x in self.available_actions

    def __repr__(self):
        return "Dynamic(%d)" % self.n

    def __eq__(self, other):
        return self.n == other.n and self.available_actions == other.available_actions