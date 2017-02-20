from collections import deque
from math import sqrt
from pprint import pprint

import numpy as np


class Board:
    def __init__(self, state=None, tiles: str = None):
        if state:
            self._board_size = len(state)
            self._sorted_tiles = np.sort(state.flatten())
            self._state = state
        else:
            starting_tiles = np.array(tiles.split(','))
            self._board_size = int(sqrt(len(starting_tiles)))
            self._sorted_tiles = np.sort(starting_tiles)
            self._state = np.reshape(starting_tiles,
                                     (self._board_size, -1))
        self._goal = np.reshape(self._sorted_tiles, (self._board_size, -1))
        self._hole = '0'

    @property
    def goal(self):
        return self._goal

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def state_string(self):
        return np.array2string(self._state.flatten())

    @property
    def goal_string(self):
        return np.array2string(self._goal.flatten())

    def hole_pos(self):
        pos = np.where(self._state == self._hole)
        return pos[0][0], pos[1][0]

    def tile(self, pos):
        if pos[0] < 0 or pos[1] < 0:
            raise ValueError('Tile out of bounds!')
        return self._state[pos]

    def neighbors(self):
        """
        find neighboring tiles to hole position
        """
        hole = self.hole_pos()
        up = (hole[0] - 1, hole[1])
        down = (hole[0] + 1, hole[1])
        left = (hole[0], hole[1] - 1)
        right = (hole[0], hole[1] + 1)
        return [('UP', up),
                ('DOWN', down),
                ('LEFT', left),
                ('RIGHT', right)]

    def swap(self, pos: tuple = None):
        """
        position is tuple (R,C) of neighboring tile to hole
        """
        try:
            hole = self.hole_pos()
            tile = self.tile(pos)
            temp_state = np.array(self._state, copy=True)
            temp_state[pos[0]][pos[1]] = self._hole
            temp_state[hole[0]][hole[1]] = tile
            return True, temp_state
        except ValueError:
            return False, self._state


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
    board = Board(tiles="3,1,2,0,4,5,6,7,8")
    # board = Board(tiles="0,1,2,3,4,5,6,7,8")
    search = BFS(board)
    node = search.solve()
    print(node)
