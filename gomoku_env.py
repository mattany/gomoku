from typing import Tuple

import gym
import numpy as np

from DynamicSpace import GomokuObservationSpace, GomokuActionSpace
from gym.utils import seeding

TIE = 9

PLAYER1 = 1

PLAYER2 = -1

WIN = -100
TIE_REWARD = 0
board_domination_heuristic = np.ndarray.flatten(np.asarray([
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
]))

board_domination_heuristic_matrix = np.asarray([
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

    def __init__(self):
        self.board = [[0 for i in range(15)] for j in range(15)]
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        return self.observation_space.numerical_representation

    def step(self, action: int, player):
        assert self.action_space.contains(action)
        self.action_space.make_move(action, player)
        reward = 0
        done = False
        status = self.check()
        if self.check():
            done = True
            if status == TIE:
                reward = TIE_REWARD
            else:
                reward = WIN
        if not done:
            # reward = sum((sum(i) for i in board_domination_heuristic_matrix * np.asarray(self.observation_space.board)))
            reward = self.longest_streak_heuristic(player)
            # if player == PLAYER1:
            #     reward = -reward
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
        is_tie = TIE
        for i in range(15):
            for j in range(15):
                if board[i][j] == 0:
                    is_tie = 0
                    continue
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
        return is_tie

    def longest_streak_heuristic(self, player):
        my_count, op_count = 0, 0
        board = self.board
        dirs = ((1, -1), (1, 0), (1, 1), (0, 1))
        for i in range(15):
            for j in range(15):
                if board[i][j] == 0:
                    is_tie = 0
                    continue
                id = board[i][j]
                for d in dirs:
                    x, y = j, i
                    count = 0
                    for k in range(5):
                        if self.get(y, x) != id: break
                        y += d[0]
                        x += d[1]
                        count += 1
                    if id == player:
                        if count > my_count:
                            my_count = count
                    else:
                        if count > op_count:
                            op_count = count
        return -20 * my_count + 20 * op_count
