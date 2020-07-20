from typing import Tuple

import gym
import numpy as np

from DynamicSpace import GomokuObservationSpace, GomokuActionSpace
from gym.utils import seeding

PLAYER1 = 1

PLAYER2 = -1

WIN = 100
TIE = 0
LOSE = -100
board_domination_heuristic = np.asarray([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0],
        [0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0],
        [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1, 0],
        [0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 4, 3, 2, 1, 0],
        [0, 1, 2, 3, 4, 5, 6, 6, 6, 5, 4, 3, 2, 1, 0],
        [0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0],
        [0, 1, 2, 3, 4, 5, 6, 6, 6, 5, 4, 3, 2, 1, 0],
        [0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 4, 3, 2, 1, 0],
        [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1, 0],
        [0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0],
        [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])





class GomokuEnv(gym.Env):

    def __init__(self, board):
        self.board = board
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        return self.observation_space.numerical_representation

    def step(self, action: int):
        assert self.action_space.contains(action)
        self.action_space.make_move(action, PLAYER1)
        reward = 0
        done = False
        if self.check():
            done = True
            reward = WIN
        else:
            opponent_action = self.action_space.sample()
            if opponent_action == -1:
                done = True
                reward = TIE
            else:
                self.action_space.make_move(opponent_action, PLAYER2)
                if self.check():
                    done = True
                    reward = LOSE
                elif self.action_space.sample() == -1:
                    done = True
                    reward = TIE
        if not done:
            reward = sum((sum(i) for i in board_domination_heuristic * np.asarray(self.observation_space.board)))
        return self._get_obs(), float(reward), done, {}

    def reset(self):
        self.board = [[0 for i in range(15)] for j in range(15)]
        self.action_space = GomokuActionSpace(self.board)
        self.observation_space = GomokuObservationSpace(self.board)
        return self.observation_space.numerical_representation

    def render(self):
        return self._get_obs().__repr__()

    def get(self, row, col):
        if row < 0 or row >= 15 or col < 0 or col >= 15:
            return 0
        return self.board[row][col]

    def check(self):
        board = self.board
        dirs = ((1, -1), (1, 0), (1, 1), (0, 1))
        for i in range(15):
            for j in range(15):
                if board[i][j] == 0: continue
                id = board[i][j]
                for d in dirs:
                    x, y = j, i
                    count = 0
                    for k in range(5):
                        if self.get(y, x) != id: break
                        y += d[0]
                        x += d[1]
                        count += 1
                    if count == 5:
                        self.won = {}
                        r, c = i, j
                        for z in range(5):
                            self.won[(r, c)] = 1
                            r += d[0]
                            c += d[1]
                        return id
        return 0