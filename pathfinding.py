"""Pathfinding search algorithms display.
Breadth first
Lowest cost
Greedy best first
A* with at least two different heuristics.

Author: Mikayla Winant
Date: 05/15/2024
"""

from functions import (
    read_map_from_file, find_start_position, find_end_position, find_barriers,
    display_map, bfs, lowest_cost_search, a_star_search
)
map_file = 'map.txt'
maze = read_map_from_file(map_file)

width: int = len(maze[0])
height: int = len(maze)
start = find_start_position(maze)
goal = find_end_position(maze)
barriers = find_barriers(maze)

print('\nBreadth First Search Map\n')
bfs_result = bfs(start, goal, barriers, width, height)
display_map(bfs_result, maze)

print('\n\n\n')

print("Lowest Cost First Map: \n")
lcs_result = lowest_cost_search(start, goal, barriers, width, height)
display_map(lcs_result, maze)

print('\n\n\n')
print("A* with Manhattan distance: \n")
# A* search with Manhattan distance heuristic
result_manhattan = a_star_search(start, goal, barriers, width, height, 'manhattan')
display_map(result_manhattan, maze)

print('\n\n\n')
print("A* with Euclidean distance: \n")
# A* search with Euclidean distance heuristic
result_euclidean = a_star_search(start, goal, barriers, width, height, 'euclidean')
display_map(result_euclidean, maze)