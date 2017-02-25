from argparse import ArgumentParser
from resource import getrusage, RUSAGE_SELF

from src.puzzle import Puzzle

if __name__ == "__main__":
    try:
        parser = ArgumentParser()
        parser.add_argument("solver", help="algorithm (bfs | dfs)")
        parser.add_argument("board", help="board string (0,1,2,3...)")
        args = parser.parse_args()
        puzzle = Puzzle(tiles=args.board, algorithm=args.solver)
        results = puzzle.solve()
        with open('output.txt', 'w+') as f:
            f.write(f"path_to_goal: {puzzle.actions}\n")
            f.write(f"cost_of_path: {puzzle.path_cost}\n")
            f.write(f"nodes_expanded: {puzzle.nodes_expanded}\n")
            f.write(f"fringe_size: {puzzle.fringe_size}\n")
            f.write(f"max_fringe_size: {puzzle.max_fringe_size}\n")
            f.write(f"search_depth: {puzzle.search_depth}\n")
            f.write(f"max_search_depth: {puzzle.max_search_depth}\n")
            f.write(f"running_time: {puzzle.running_time}\n")
            f.write(f"max_ram_usage: {getrusage(RUSAGE_SELF)[2]/(1024**2)}\n")
    except TypeError as e:
        print(e)
        exit(1)
    except RuntimeError as e:
        print(e)
        exit(1)
