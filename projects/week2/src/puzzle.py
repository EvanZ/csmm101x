from argparse import ArgumentParser
from resource import getrusage, RUSAGE_SELF

from board import Board
from solver import BFS, DFS, AST

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
        with open('output.txt', 'w+') as f:
            f.write(f"path_to_goal: {res.actions}\n")
            f.write(f"cost_of_path: {res.path_cost}\n")
            f.write(f"nodes_expanded: {search.nodes_expanded}\n")
            f.write(f"fringe_size: {search.fringe_size}\n")
            f.write(f"max_fringe_size: {search.max_fringe_size}\n")
            f.write(f"search_depth: {search.search_depth}\n")
            f.write(f"max_search_depth: {search.max_search_depth}\n")
            f.write(f"running_time: {search.running_time}\n")
            f.write(f"max_ram_usage: {getrusage(RUSAGE_SELF)[2]/(1024**2)}\n")
    except TypeError as e:
        print(e)
        exit(1)
    except RuntimeError as e:
        print(e)
        exit(1)
