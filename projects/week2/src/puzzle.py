import abc
import heapq
from argparse import ArgumentParser
from collections import deque
from math import sqrt
from resource import getrusage, RUSAGE_SELF
from time import time

import numpy as np


class Board:
    def __init__(self, tiles: str, hole: str = '0', sz: int = None):
        board_ = tiles.split(',')
        self._sz = sz or int(sqrt(len(board_)))
        self._sorted_tokens = [str(x) for x in range(self._sz ** 2)]
        # self._goal = np.reshape(self._sorted_tokens, (self._sz, -1))
        self._goal = [[str(i + j * self._sz) for i in range(self._sz)] for j in range(self._sz)]
        self._state = np.reshape(board_, (self._sz, -1))
        # self._state = [[board_[i + j * self._sz] for i in range(self._sz)] for j in range(self._sz)]
        self._hole = hole
        # self._hole_pos = self.hole_pos()
        self._valid_moves = self.valid_moves()
        self._neighbors = self.neighbors()

    def __repr__(self):
        return str(self._state)

    @property
    def actions(self):
        return self._valid_moves

    @property
    def size(self):
        return self._sz

    @property
    def goal(self):
        return ','.join(self._sorted_tokens)

    @property
    def sorted_tokens(self):
        return self._sorted_tokens

    @property
    def string(self):
        return self.stringify(self._state)

    @property
    def state(self):
        return self._state

    @staticmethod
    def stringify(state):
        # return ','.join(state.flatten())
        stringified = ','.join([item for sublist in state for item in sublist])
        return stringified

    def hole_pos(self):
        # pos = np.where(self._state == self._hole)
        # return pos[0][0], pos[1][0]
        ix = [self._state[r][c] == self._hole for r in range(self._sz)
              for c in range(self._sz)].index(True)
        hole_pos_ = divmod(ix, self._sz)
        return hole_pos_

    def tile(self, pos):
        return self._state[pos[0]][pos[1]]

    def neighbors(self):
        r, c = self.hole_pos()
        neighbors = {
            'Up': (r - 1, c),
            'Down': (r + 1, c),
            'Left': (r, c - 1),
            'Right': (r, c + 1)
        }
        return neighbors

    def act(self, action):
        pos = self._neighbors[action]
        board_ = self.swap(pos)
        return board_

    def valid_moves(self):
        """
        find neighboring tiles to hole position
        """
        hole = self.hole_pos()
        actions_ = []
        if hole[0] > 0:
            actions_.append('Up')
        if hole[0] < self._sz - 1:
            actions_.append('Down')
        if hole[1] > 0:
            actions_.append('Left')
        if hole[1] < self._sz - 1:
            actions_.append('Right')
        return actions_

    def swap(self, pos):
        """
        position is tuple (R,C) of neighboring tile to hole
        """
        hole = self.hole_pos()
        tile = self.tile(pos)
        temp_state = self._state.copy()
        temp_state[pos[0]][pos[1]] = self._hole
        temp_state[hole[0]][hole[1]] = tile
        new_board = Board(self.stringify(temp_state),
                          hole=self._hole,
                          sz=self._sz)
        return new_board


class Node:
    def __init__(self, state, action=None, path_cost=None, parent=None):
        if not parent:
            self._state = state
            self._actions = []
            self._path_costs = []
        else:
            self._state = state
            self._actions = parent.actions[:]
            self._actions.append(action)
            self._path_costs = parent.path_costs[:]
            self._path_costs.append(path_cost)

    def __repr__(self):
        return str({'state': self._state.state,
                    'action': self._actions,
                    'path_cost': self.path_cost})

    @property
    def state(self):
        return self._state

    @property
    def actions(self):
        return self._actions

    @property
    def path_costs(self):
        return self._path_costs

    @property
    def path_cost(self):
        return sum(self._path_costs)

    @property
    def depth(self):
        return len(self._actions)  # to account for 0 indexing


class Solver(metaclass=abc.ABCMeta):
    def __init__(self, start_board: Board, depth: int = None):
        self._goal = start_board.goal
        self._start_board = start_board
        self._frontier = deque()
        self._explored = set()
        self._path_cost = 0
        self._nodes_expanded = 0
        self._fringe_sz = 0
        self._max_fringe_sz = 0
        self._search_depth = 0
        self._max_search_depth = 0
        self._running_time = 0
        self._depth_limit = depth

    @property
    def nodes_expanded(self):
        return self._nodes_expanded

    @property
    def fringe_size(self):
        return self._fringe_sz

    @property
    def max_fringe_size(self):
        return self._max_fringe_sz

    @property
    def search_depth(self):
        return self._search_depth

    @property
    def max_search_depth(self):
        return self._max_search_depth

    @property
    def running_time(self):
        return self._running_time

    def update_fringe_size(self):
        self._fringe_sz = len(self._frontier)
        if self._fringe_sz > self._max_fringe_sz:
            self._max_fringe_sz = self._fringe_sz

    @abc.abstractmethod
    def solve(self):
        """
        To be implemented by detailed search strategies
        """
        return


class AST(Solver):
    def __init__(self, start_board: Board, depth: int = None):
        super().__init__(start_board, depth)
        self._frontier = []
        self.action_priority = {
            'Up': 0,
            'Down': 1,
            'Left': 2,
            'Right': 3
        }

    @staticmethod
    def h(state):
        cost = 0
        for r in range(state.size):
            for c in range(state.size):
                tok = int(state.tile((r, c)))
                r_goal, c_goal = divmod(tok, state.size)
                cost += abs(r_goal - r) + abs(c_goal - c)
        return cost

    def solve(self):
        start_time = time()
        root = Node(state=self._start_board)
        self._frontier.append((root.path_cost, None, 0, root))
        heapq.heapify(self._frontier)
        if root.state.string == self._goal:
            self._search_depth = root.depth
            self._running_time = time() - start_time
            return root
        while True:
            if len(self._frontier) == 0:
                self._running_time = time() - start_time
                raise ValueError('Goal not found.')
            cost, action, r, node = heapq.heappop(self._frontier)
            if node.state.string == self._goal:
                self.update_fringe_size()
                self._search_depth = node.depth
                self._running_time = time() - start_time
                return node
            self._nodes_expanded += 1
            self._explored.add(node.state.string)
            actions = node.state.actions
            for index, action in enumerate(actions):
                state = node.state.act(action)
                child = Node(state=state,
                             action=action,
                             path_cost=1,
                             parent=node)
                if child.depth > self._max_search_depth:
                    self._max_search_depth = child.depth
                if child.state.string not in self._explored:
                    heapq.heappush(self._frontier,
                                   (child.path_cost + self.h(state),
                                    self.action_priority[action],
                                    np.random.rand(),
                                    child))
                    print(child.path_cost + self.h(state), action, state)
                    self._explored.add(child.state.string)
                    self.update_fringe_size()


class IDA(Solver):
    def solve(self):
        pass


class BFS(Solver):
    def solve(self):
        start_time = time()
        root = Node(state=self._start_board)
        self._frontier.append(root)
        if root.state.string == self._goal:
            self._search_depth = root.depth
            self._running_time = time() - start_time
            return root
        print(root)
        while True:
            if len(self._frontier) == 0:
                self._running_time = time() - start_time
                raise ValueError('Goal not found.')
            node = self._frontier.popleft()
            if node.state.string == self._goal:
                self.update_fringe_size()
                self._search_depth = node.depth
                self._running_time = time() - start_time
                return node
            self._nodes_expanded += 1
            self._explored.add(node.state.string)
            actions = node.state.actions
            for action in actions:
                state = node.state.act(action)
                child = Node(state=state,
                             action=action,
                             path_cost=1,
                             parent=node)
                if child.depth > self._max_search_depth:
                    self._max_search_depth = child.depth
                if child.state.string not in self._explored:
                    self._frontier.append(child)
                    self._explored.add(child.state.string)
                    self.update_fringe_size()


class DFS(Solver):
    def solve(self):
        start_time = time()
        root = Node(state=self._start_board)
        self._frontier.append(root)
        if root.state.string == self._goal:
            self._search_depth = root.depth
            self._running_time = time() - start_time
            return root
        while True:
            if len(self._frontier) == 0:
                self._running_time = time() - start_time
                raise ValueError('Goal not found.')
            node = self._frontier.pop()
            self._explored.add(node.state.string)
            if node.state.string == self._goal:
                self.update_fringe_size()
                self._search_depth = node.depth
                self._running_time = time() - start_time
                return node
            self._nodes_expanded += 1
            actions = node.state.actions
            for action in list(reversed(actions)):
                state = node.state.act(action)
                child = Node(state=state,
                             action=action,
                             path_cost=1,
                             parent=node)
                print(child)
                if child.depth > self._max_search_depth:
                    self._max_search_depth = child.depth
                if child.state.string not in self._explored:
                    self._frontier.append(child)
                    self._explored.add(child.state.string)
                    self.update_fringe_size()


if __name__ == "__main__":
    try:
        parser = ArgumentParser()
        parser.add_argument("solver", help="algorithm (bfs | dfs)")
        parser.add_argument("board", help="board string (0,1,2,3...)")
        args = parser.parse_args()
        board = Board(tiles=args.board)
        print("***STARTING STATE***")
        print(board.state)
        algorithms = {
            'bfs': BFS(board),
            'dfs': DFS(board),
            'ast': AST(board),
            # 'ida': BFS(board, depth=100)
        }

        search = algorithms[args.solver]
        res = search.solve()
        print("***SOLUTION STATE***")
        print(res)
        print(f"nodes expanded {search.nodes_expanded}")
        print(f"path cost {res.path_cost}")
        print(f"actions {res.actions}")
        print(f"fringe size: {search.fringe_size}")
        print(f"max_fringe_size: {search.max_fringe_size}")
        print(f"search depth {search.search_depth}")
        print(f"max search depth {search.max_search_depth}")
        print(f"running time {search.running_time}")
        print(f"memory usage {getrusage(RUSAGE_SELF)[2]}")
    except TypeError as e:
        print(e)
        exit(1)
    except RuntimeError as e:
        print(e)
        exit(1)
