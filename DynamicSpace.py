import gym
from gym.spaces import Discrete
import numpy as np
import math


# You could also inherit from Discrete or Box here and just override the shape(), sample() and contains() methods
class GomokuObservationSpace(Discrete):
    """
    x where x in available actions {0,1,3,5,...,n-1}
    Example usage:
    self.action_space = spaces.Dynamic(max_space=2)
    """

    def __init__(self, board, size = 15):
        cells = 15**2
        super().__init__(3**cells)
        self.board = board

    # def enable_actions(self, row, col):
    #     """ You would call this method inside your environment to enable actions"""
    #     self.available_actions[row][col] = 0
    #     return self.available_actions

    def sample(self):
        illegal = True
        row, col = None, None
        if any(any((i == 0) for i in j) for j in self.board):
            while illegal:
                row = np.random.randint(len(self.board))
                col = np.random.randint(len(self.board[row]))
                illegal = self.board[row][col]
            return row, col
        return -1

    def contains(self, x):
        return not self.board[x[0]][x[1]]

    @property
    def numerical_representation(self):
        retval = int(''.join([''.join([str(i + 1) for i in j]) for j in self.board]))
        return convertToDecimal(retval)

    def __repr__(self):
        return np.asarray(self.board).__repr__()

    def __eq__(self, other):
        return self.board == other.board



class GomokuActionSpace(Discrete):
    """
    x where x in available actions {0,1,3,5,...,n-1}
    Example usage:
    self.action_space = spaces.Dynamic(max_space=2)
    """

    def __init__(self, board, size=15):
        super().__init__(size ** 2)
        # initially all actions are available

        self.board = board

    def make_move(self, action, val):
        """ You would call this method inside your environment to remove available actions"""

        div = action // 15
        self.board[div][action - (div * 15)] = val

    # def enable_actions(self, row, col):
    #     """ You would call this method inside your environment to enable actions"""
    #     self.available_actions[row][col] = 0
    #     return self.available_actions

    def sample(self):
        illegal = True
        row, col = None, None
        if any(any((i == 0) for i in j) for j in self.board):
            while illegal:
                row = np.random.randint(len(self.board))
                col = np.random.randint(len(self.board[row]))
                illegal = self.board[row][col]
            return row * 15 + col
        return -1

    def contains(self, x):

        div = x // 15
        return not self.board[div][x - (div * 15)]

    def __repr__(self):
        return np.asarray(self.board).__repr__()

    def __eq__(self, other):
        return self.board == other.board


def convertToTernary(N):
    # Base case
    if (N == 0):
        return

        # Finding the remainder
    # when N is divided by 3
    x = N % 3
    N //= 3
    if (x < 0):
        N += 1

        # Recursive function to
    # call the function for
    # the integer division
    # of the value N/3
    convertToTernary(N)

    # Handling the negative cases
    if (x < 0):
        print(x + (3 * -1), end="")
    else:
        print(x, end="")



def convertToDecimal(N):

    # If the number is greater than 0,
    # compute the decimal
    # representation of the number
    if (N != 0):
        decimalNumber = 0
        i = 0
        remainder = 0

        # Loop to iterate through
        # the number
        while (N != 0):
            remainder = N % 10
            N = N // 10

            # Computing the decimal digit
            decimalNumber += int(remainder * math.pow(3, i))
            i += 1

        return decimalNumber

    else:
        return 0
