from collections import deque
import copy
from math import sqrt
from pprint import pprint

import numpy as np


class Board:
    def __init__(self, tiles: str, sz: int, hole: str = '0'):
        self._sz = sz
        self._sorted_tokens = [str(x) for x in range(sz ** 2)]
        self._goal = np.reshape(self._sorted_tokens, (sz, -1))
        self._state = np.reshape(tiles.split(','), (sz, -1))
        self._hole = hole

    def __repr__(self):
        return str(self._state)

    @property
    def num_tiles(self):
        return self._sz ** 2

    @property
    def num_rows(self):
        return self._sz

    @property
    def goal(self):
        return ','.join(self._sorted_tokens)

    @property
    def string(self):
        return self.stringify(self._state)

    @property
    def state(self):
        return self._state

    @staticmethod
    def stringify(state):
        return ','.join(state.flatten())

    def hole_pos(self):
        pos = np.where(self._state == self._hole)
        return pos[0][0], pos[1][0]

    def tile(self, pos):
        if pos[0] < 0 or pos[1] < 0:
            raise ValueError('Tile out of bounds!')
        return self._state[pos]

    def actions(self):
        """
        find neighboring tiles to hole position
        """
        hole = self.hole_pos()
        actions_ = []
        if hole[0] - 1 >= 0:
            actions_.append(('U',
                             (hole[0] - 1, hole[1])))
        if hole[0] + 1 < self._sz:
            actions_.append(('D',
                             (hole[0] + 1, hole[1])))
        if hole[1] - 1 >= 0:
            actions_.append(('L',
                             (hole[0], hole[1] - 1)))
        if hole[1] + 1 < self._sz:
            actions_.append(('R',
                             (hole[0], hole[1] + 1)))
        return actions_

    def swap(self, pos):
        """
        position is tuple (R,C) of neighboring tile to hole
        """
        try:
            hole = self.hole_pos()
            tile = self.tile(pos)
            temp_state = np.array(self._state, copy=True)
            temp_state[pos[0]][pos[1]] = self._hole
            temp_state[hole[0]][hole[1]] = tile
            return Board(self.stringify(temp_state),
                         sz=self._sz,
                         hole=self._hole)
        except ValueError:
            return None


class Node:
    def __init__(self, state=None, action=None, path_cost=None, parent=None):
        self._state = state
        self._action = action
        self._path_cost = path_cost
        self._parent = parent

    def __repr__(self):
        return str({'state': self._state.state,
                    'action': self._action,
                    'path_cost': self._path_cost,
                    'parent': self.parent})

    @property
    def state(self):
        return self._state

    @property
    def action(self):
        return self._action

    @property
    def path_cost(self):
        return self._path_cost

    @property
    def parent(self):
        return self._parent


class BFS:
    def __init__(self, start_board):
        self._goal = start_board.goal_string
        self._start_board = start_board
        self._frontier = deque()
        self._explored = set()
        self._path_cost = 0

    def solve(self):
        root = Node(state=self._start_board,
                    path_cost=self._path_cost)
        self._frontier.append(root)
        if root.state.state_string == self._goal:
            return root
        while True:
            if len(self._frontier) == 0:
                raise ValueError('Goal not found.')
            node = self._frontier.popleft()
            pprint(node)
            self._explored.add(node.state.state_string)
            print(self._explored)
            actions = node.state.neighbors()
            print(actions)
            for action in actions:
                res, new_state = node.state.swap(action[1])
                print(res, new_state)
                if res:
                    child = Node(state=Board(state=new_state),
                                 action=node.state.action[0],
                                 path_cost=1,
                                 parent=node)


if __name__ == "__main__":
    tiles = "0,1,2,3"
    board = Board(tiles=tiles, sz=2)
    print(board)
    print(board.stringify(board.state))
    actions = board.actions()
    print(actions)
    for action in actions:
        print(action[0])
        temp = board.swap(action[1])
        print(temp)
