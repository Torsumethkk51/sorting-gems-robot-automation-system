# ========== A* PATHFINDING ==========

import heapq
import math
import numpy as np

class Node:
    def __init__(self, pos, parent=None, g=0.0, h=0.0):
        # pos is stored as (grid_x, grid_y)
        self.pos = pos
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b):
    # euclidean distance heuristic
    return math.hypot(b[0] - a[0], b[1] - a[1])

def astar_search(grid: np.ndarray, start_px: tuple, goal_px: tuple, step_size: int = 15):
    # 1.downscale pixel coordinates to grid coordinates
    start_grid = (int(round(start_px[0] / step_size)), int(round(start_px[1] / step_size)))
    goal_grid = (int(round(goal_px[0] / step_size)), int(round(goal_px[1] / step_size)))

    gh, gw = grid.shape
    max_gx = int(gw // step_size)
    max_gy = int(gh // step_size)

    # 2.initialize open and closed sets
    open_heap = []
    start_node = Node(start_grid, None, 0.0, heuristic(start_grid, goal_grid))
    heapq.heappush(open_heap, start_node)
    
    # dictionary to keep track of lowest g cost found so far for each grid cell
    visited_g = {start_grid: 0.0}

    # 8-directional motion vectors: (dx, dy, step_cost)
    directions = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
    ]

    found_node = None

    # 3.run a* loop
    while open_heap:
        current = heapq.heappop(open_heap)

        # check if goal reached
        if current.pos == goal_grid:
            found_node = current
            break

        # skip node if we already discovered a cheaper route to it
        if current.g > visited_g.get(current.pos, float('inf')):
            continue

        for dx, dy, move_cost in directions:
            nx = current.pos[0] + dx
            ny = current.pos[1] + dy

            # check boundary limits
            if not (0 <= nx < max_gx and 0 <= ny < max_gy):
                continue

            # check obstacle collision on inflated grid
            px = int(nx * step_size)
            py = int(ny * step_size)
            if grid[py, px] > 0:
                continue

            new_g = current.g + move_cost
            next_pos = (nx, ny)

            if new_g < visited_g.get(next_pos, float('inf')):
                visited_g[next_pos] = new_g
                h_cost = heuristic(next_pos, goal_grid)
                neighbor_node = Node(next_pos, current, new_g, h_cost)
                heapq.heappush(open_heap, neighbor_node)

    # 4.reconstruct path from target to start
    if found_node is None:
        return []

    grid_path = []
    curr = found_node
    while curr is not None:
        grid_path.append(curr.pos)
        curr = curr.parent
    grid_path.reverse()

    # 5.convert grid path back into full-resolution pixel coordinates
    pixel_path = [(int(gx * step_size), int(gy * step_size)) for gx, gy in grid_path]
    return pixel_path

def prune_waypoints(path: list, min_dist: float = 35.0):
    # simplify dense path into sparser waypoints for smooth differential drive
    if len(path) <= 2:
        return path

    pruned = [path[0]]
    for pt in path[1:-1]:
        prev = pruned[-1]
        if math.hypot(pt[0] - prev[0], pt[1] - prev[1]) >= min_dist:
            pruned.append(pt)
    pruned.append(path[-1])
    return pruned

# ====================================