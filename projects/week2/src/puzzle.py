from collections import deque

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
            temp_state = self._state.copy()
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

    def __iter__(self):
        node = self
        while node:
            yield node
            node = node._parent

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
    def __init__(self, start_board: Board):
        self._goal = start_board.goal
        self._start_board = start_board
        self._frontier = deque()
        self._explored = set()
        self._path_cost = 0
        self._nodes_expanded = 0
        self._fringe_sz = 0
        self._max_fringe_sz = 0

    @property
    def nodes_expanded(self):
        return self._nodes_expanded

    @property
    def fringe_size(self):
        return self._fringe_sz

    @property
    def max_fringe_size(self):
        return self._max_fringe_sz

    def update_fringe_size(self):
        self._fringe_sz = len(self._frontier)
        if self._fringe_sz > self._max_fringe_sz:
            self._max_fringe_sz = self._fringe_sz

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
            if node.state.string == self._goal:
                self.update_fringe_size()
                return node
            self._nodes_expanded += 1
            self._explored.add(node.state.string)
            actions = node.state.actions()
            for action in actions:
                state = node.state.act(action)
                child = Node(state=state,
                             action=action,
                             path_cost=1,
                             parent=node)
                if (child.state.string not in self._explored) and (child not in self._frontier):
                    self._frontier.append(child)
                    self.update_fringe_size()


class Summary:
    def __init__(self, child: Node):
        self._child = child

    def path_cost(self):
        return sum(n.path_cost for n in self._child)

    def actions(self):
        return list(reversed([n.action for n in self._child]))[1:]


if __name__ == "__main__":
    init_tiles = "1,2,5,3,4,0,6,7,8"
    board = Board(tiles=init_tiles, sz=3)
    bfs = BFS(board)
    res = bfs.solve()
    print(f"nodes expanded {bfs.nodes_expanded}")
    summary = Summary(res)
    print(f"path cost {summary.path_cost()}")
    print(summary.actions())
    print(f"fringe size: {bfs.fringe_size}")
    print(f"max_fringe_size: {bfs.max_fringe_size}")
