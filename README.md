# csmm101x
edX course on AI

## Week 2 project
Implement 8-puzzle solver with BFS, DFS, A*, and IDA*

I use 3 classes to implement solutions:
* Board contains puzzle state and methods for manipulating tiles
* Node contains a Board and path of actions leading to that state (i.e.
a tree) along with path costs
* Solver is an abstract base class for each type of solver (concrete class
must implement solve method)

Command line execution (solver can be <code>ast</code> |
<code>bfs</code> | <code>dfs</code> | <code>ida</code>):
<code>
python3 driver.py ast 1,2,5,3,4,0,6,7,8
</code>