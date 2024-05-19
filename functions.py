"""Functions used in pathfinding algorithms.

Author: Mikayla Winant
Date: 05/15/2024
"""
import heapq
import math
from collections import deque, namedtuple
from typing import List, Tuple, Set

# flake8: noqa
SearchResult = namedtuple('SearchResult', ['path', 'explored', 'open_list'])


def read_map_from_file(file_name):
    """Convert text file into map."""
    with open(file_name, 'r', encoding='utf8') as file:
        return [list(line.strip()) for line in file]


def find_start_position(maze):
    """Find start position on map marked with 'S'."""
    for y, row in enumerate(maze):
        for x, char in enumerate(row):
            if char == 'S':
                return (x, y)
    return None


def find_end_position(maze):
    """Find end position on map marked with 'E'."""
    for y, row in enumerate(maze):
        for x, char in enumerate(row):
            if char == 'E':
                return (x, y)
    return None


def find_barriers(maze):
    """Find barriers on map marked with 'X'."""
    barriers = []
    for y, row in enumerate(maze):
        for x, char in enumerate(row):
            if char == 'X':
                barriers.append((x, y))
    return barriers


def display_map(result, original_map):
    """Display map."""
    map_copy = [row[:] for row in original_map]

    path, explored, open_list = result.path, result.explored, result.open_list

    # Draw path on map copy
    for x, y in path:
        map_copy[y][x] = 'P'

    # Mark explored squares on the map copy
    for x, y in explored:
        if map_copy[y][x] != 'P':
            map_copy[y][x] = '.'

    # Mark the current open list on the map copy
    for x, y in open_list:
        if map_copy[y][x] != 'P' and map_copy[y][x] != '.':
            map_copy[y][x] = 'O'

    # Mark start and end positions on the map copy
    start_x, start_y = result.path[0]
    end_x, end_y = result.path[-1]
    map_copy[start_y][start_x] = 'S'
    map_copy[end_y][end_x] = 'E'

    # Print the final map copy
    for row in map_copy:
        print(' '.join(row))


def bfs(start: Tuple[int, int], goal: Tuple[int, int], barriers: Set[Tuple[int, int]], width: int, height: int) -> SearchResult:
    """Breadth First Search Algorithm."""
    queue = deque([start])
    parent = {start: None}
    explored = set()
    open_list = set()
    open_list.add(start)

    while queue:
        current = queue.popleft()
        open_list.remove(current)
        explored.add(current)

        if current == goal:
            break

        x, y = current
        neighbors = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]  # Only up, down, left, right

        for neighbor in neighbors:
            if neighbor[0] < 0 or neighbor[0] >= width or neighbor[1] < 0 or neighbor[1] >= height:
                continue
            if neighbor in barriers or neighbor in explored:
                continue
            if neighbor not in open_list:
                queue.append(neighbor)
                open_list.add(neighbor)
                parent[neighbor] = current

    # Reconstruct the path
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()

    return SearchResult(path=path, explored=explored, open_list=open_list)


def lowest_cost_search(start, goal, barriers, width, height):
    """Dijkstra's Algorithm for Lowest Cost Path."""
    queue = [(0, start)]
    parent = {start: None}
    cost_so_far = {start: 0}
    closed_list = set()

    while queue:
        current_cost, current = heapq.heappop(queue)

        if current in closed_list:
            continue
        closed_list.add(current)

        if current == goal:
            break

        x, y = current
        neighbors = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]  # Only up, down, left, right

        for neighbor in neighbors:
            if neighbor[0] < 0 or neighbor[0] >= width or neighbor[1] < 0 or neighbor[1] >= height:
                continue
            if neighbor in barriers:
                continue
            new_cost = current_cost + 1  # Assuming each step has a cost of 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost
                heapq.heappush(queue, (priority, neighbor))
                parent[neighbor] = current

    # Reconstruct the path
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()

    return SearchResult(path=path, explored=set(cost_so_far.keys()), open_list=set())


def manhattan_distance(start, goal):
    """Calculate Manhattan distance heuristic."""
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])


def greedy_best_first_search(start, goal, barriers, width, height):
    """Greedy Best First Search Algorithm without diagonal movement."""
    queue = [(manhattan_distance(start, goal), start)]
    parent = {start: None}
    closed_list = set()
    open_list = set([start])

    while queue:
        _, current = heapq.heappop(queue)

        if current in closed_list:
            continue

        closed_list.add(current)
        open_list.remove(current)

        if current == goal:
            break

        x, y = current
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

        for neighbor in neighbors:
            if neighbor[0] < 0 or neighbor[0] >= width or neighbor[1] < 0 or neighbor[1] >= height:
                continue
            if neighbor in barriers:
                continue
            if neighbor not in parent:
                priority = manhattan_distance(neighbor, goal)
                heapq.heappush(queue, (priority, neighbor))
                parent[neighbor] = current
                if neighbor not in closed_list:
                    open_list.add(neighbor)

    # Reconstruct the path
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()

    return SearchResult(path=path, explored=closed_list, open_list=open_list)


def euclidean_distance(start, goal):
    """Calculate Euclidean distance heuristic."""
    return math.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2)


def a_star_search(start, goal, barriers, width, height, heuristic='manhattan'):
    """A* Search Algorithm without diagonal movement."""
    if heuristic == 'manhattan':
        h = manhattan_distance
    elif heuristic == 'euclidean':
        h = euclidean_distance
    else:
        raise ValueError("Unknown heuristic")

    queue = [(0 + h(start, goal), 0, start)]  # (f, g, position)
    parent = {start: None}
    cost_so_far = {start: 0}
    closed_list = set()
    open_list = set([start])

    while queue:
        _, g, current = heapq.heappop(queue)

        if current in closed_list:
            continue

        closed_list.add(current)
        open_list.remove(current)

        if current == goal:
            break

        x, y = current
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

        for neighbor in neighbors:
            if neighbor[0] < 0 or neighbor[0] >= width or neighbor[1] < 0 or neighbor[1] >= height:
                continue
            if neighbor in barriers:
                continue
            new_cost = g + 1  # Assuming each step has a cost of 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + h(neighbor, goal)
                heapq.heappush(queue, (priority, new_cost, neighbor))
                parent[neighbor] = current
                if neighbor not in closed_list:
                    open_list.add(neighbor)

    # Reconstruct the path
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()

    return SearchResult(path=path, explored=set(cost_so_far.keys()), open_list=open_list)
