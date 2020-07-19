import gym
from DynamicSpace import Dynamic
from gym.utils import seeding


class GomokuEnv(gym.Env):

    def __init__(self):
        self.action_space = Dynamic(225)

