from collections import deque
from pprint import pprint

import numpy as np


class Board:
    def __init__(self, tiles_: str, sz: int, hole: str = '0'):
        self._sz = sz
        self._sorted_tokens = [str(x) for x in range(sz ** 2)]
        self._goal = np.reshape(self._sorted_tokens, (sz, -1))
        self._state = np.reshape(tiles_.split(','), (sz, -1))
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

    def act(self, action):
        hole = self.hole_pos()
        lut = {
            'U': (hole[0] - 1, hole[1]),
            'D': (hole[0] + 1, hole[1]),
            'L': (hole[0], hole[1] - 1),
            'R': (hole[0], hole[1] + 1)
        }
        pos = lut[action]
        board_ = self.swap(pos)
        return board_

    def actions(self):
        """
        find neighboring tiles to hole position
        """
        hole = self.hole_pos()
        actions_ = []
        if hole[0] - 1 >= 0:
            actions_.append('U')
        if hole[0] + 1 < self._sz:
            actions_.append('D')
        if hole[1] - 1 >= 0:
            actions_.append('L')
        if hole[1] + 1 < self._sz:
            actions_.append('R')
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
        self._goal = start_board.goal
        self._start_board = start_board
        self._frontier = deque()
        self._explored = set()
        self._path_cost = 0

    def solve(self):
        root = Node(state=self._start_board,
                    path_cost=self._path_cost)
        self._frontier.append(root)
        if root.state.string == self._goal:
            return root
        while True:
            if len(self._frontier) == 0:
                raise ValueError('Goal not found.')
            node = self._frontier.popleft()
            pprint(node)
            self._explored.add(node.state.string)
            print(self._explored)
            actions = node.state.actions()
            for action in actions:
                print(action)
                print(node.state)
                state = node.state.act(action)
                child = Node(state=state,
                             action=action,
                             path_cost=1,
                             parent=node)
                print(child)


if __name__ == "__main__":
    tiles = "3,1,2,0,4,5,6,7,8"
    board = Board(tiles=tiles, sz=3)
    bfs = BFS(board)
    bfs.solve()
