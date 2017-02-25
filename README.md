# csmm101x
edX course on AI

## Week 2 project
### <code>Python 3.6.</code>
[![Travis](https://img.shields.io/travis/rust-lang/rust.svg?style=flat-square)]()

Implement 8-puzzle solver with BFS, DFS, A-star, and iterative depth
A-star.

See project requirements here:

https://courses.edx.org/courses/course-v1:ColumbiaX+CSMM.101x+1T2017/courseware/ea3118a1b62849b99423f8b1182e1bbf/a1f977f8f5ab4e79a123133a94d77c7e/

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