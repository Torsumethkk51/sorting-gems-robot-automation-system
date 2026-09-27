# ========== GEM SELECTOR ==========

import math
import cv2
import numpy as np
from lib import pathfinding as pf
from lib import obstacles_detector as od_obs

def find_best_mission(robot_pos: tuple, gems_dict: dict, drop_zones: dict, static_walls: np.ndarray, step_size: int = 10):
    candidates = []
    for color_name, pos_list in gems_dict.items():
        for pos in pos_list:
            candidates.append({"color": color_name, "pos": pos})

    if not candidates:
        return None, [], [], float("inf")

    best_gem = None
    best_pickup_path = []
    best_drop_path = []
    min_total_cost = float("inf")

    # generate base collision grid once
    base_grid = od_obs.build_dynamic_grid(static_walls, {}, robot_radius_px=16)

    for gem in candidates:
        gx, gy = gem["pos"]
        gem_color = gem["color"]

        if gem_color not in drop_zones:
            continue

        drop_pos = drop_zones[gem_color]

        # copy base grid and ensure start, gem, and drop targets are completely clear
        temp_grid = base_grid.copy()
        cv2.circle(temp_grid, (int(robot_pos[0]), int(robot_pos[1])), 35, 0, -1)
        cv2.circle(temp_grid, (int(gx), int(gy)), 35, 0, -1)
        cv2.circle(temp_grid, (int(drop_pos[0]), int(drop_pos[1])), 35, 0, -1)

        # leg 1: robot -> gem
        pickup_path = pf.astar_search(temp_grid, robot_pos, (gx, gy), step_size=step_size)
        if not pickup_path:
            continue

        # leg 2: gem -> drop zone
        drop_path = pf.astar_search(temp_grid, (gx, gy), drop_pos, step_size=step_size)
        if not drop_path:
            continue

        pickup_cost = sum(math.hypot(pickup_path[i+1][0] - pickup_path[i][0], pickup_path[i+1][1] - pickup_path[i][1]) for i in range(len(pickup_path) - 1))
        drop_cost = sum(math.hypot(drop_path[i+1][0] - drop_path[i][0], drop_path[i+1][1] - drop_path[i][1]) for i in range(len(drop_path) - 1))
        total_cost = pickup_cost + drop_cost

        if total_cost < min_total_cost:
            min_total_cost = total_cost
            best_gem = gem
            best_pickup_path = pf.prune_waypoints(pickup_path, min_dist=25.0)
            best_drop_path = pf.prune_waypoints(drop_path, min_dist=25.0)

    return best_gem, best_pickup_path, best_drop_path, min_total_cost

def render_mission(image: cv2.typing.MatLike, robot_pos: tuple, best_gem: dict, drop_pos: tuple, pickup_path: list, drop_path: list, remaining_count: int):
    vis_img = image.copy()

    # 1.draw pickup path (robot to gem) in yellow
    if pickup_path:
        for i in range(len(pickup_path) - 1):
            cv2.line(vis_img, pickup_path[i], pickup_path[i+1], (0, 255, 255), 3)
        for pt in pickup_path:
            cv2.circle(vis_img, pt, 4, (0, 180, 255), -1)

    # 2.draw drop path (gem to dropzone) in cyan
    if drop_path:
        for i in range(len(drop_path) - 1):
            cv2.line(vis_img, drop_path[i], drop_path[i+1], (255, 255, 0), 2)
        for pt in drop_path:
            cv2.circle(vis_img, pt, 4, (255, 200, 0), -1)

    # 3.highlight current robot position
    cv2.circle(vis_img, (int(robot_pos[0]), int(robot_pos[1])), 10, (255, 0, 255), -1)
    cv2.putText(vis_img, "ROBOT", (int(robot_pos[0]) - 25, int(robot_pos[1]) - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # 4.highlight target gem
    if best_gem:
        gx, gy = best_gem["pos"]
        cv2.circle(vis_img, (int(gx), int(gy)), 18, (0, 255, 0), 3)

    # 5.display mission status dashboard
    cv2.rectangle(vis_img, (15, 15), (420, 110), (30, 30, 30), -1)
    cv2.rectangle(vis_img, (15, 15), (420, 110), (200, 200, 200), 1)

    gem_text = f"Target Gem: {best_gem['color'].upper()}" if best_gem else "No Gem Found"
    cv2.putText(vis_img, gem_text, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(vis_img, f"Remaining Gems: {remaining_count}", (30, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(vis_img, "Press [N] for Next Gem | [Q] to Quit", (30, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return vis_img