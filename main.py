# ========== MAIN PROGRAM ==========

import cv2
from lib import vision_cleansing as vc
from lib import aruco_tracking as at
from lib import color_filtering as cf
from lib import object_detector as od
from lib import obstacles_detector as obs
from lib import obstacles_visualization as obsv
from lib import pathfinding as pf

if __name__ == "__main__":
    image = cv2.imread("test-image/field.png")
    test_arUco_image = cv2.imread("test-image/field_aruco_1.png")
    field_with_drop_zone = cv2.imread("test-image/field_with_drop_zone_obstacle_aruco.png")

    if image is not None:
        cropped_image = vc.vision_cleansing(image=field_with_drop_zone)
        robot = at.aruco_tracker(image=cropped_image)
        scale_factor = at.get_scale_factor(image=cropped_image, physical_aruco_size=8)

        current_color_range = cf.color_filtering(image=cropped_image)

        annotated_img, drop_zones, gems = od.detect_objects(cropped_image, current_color_range)

        print("=== DROP ZONES ===")
        for color, pos in drop_zones.items():
            print(f"{color}: {pos}")

        print("\n=== GEMS ===")
        for color, pos_list in gems.items():
            print(f"{color} (amount : {len(pos_list)} gems): {pos_list}")

        robot_corners = robot["corners"] if robot is not None else None
        raw_walls, inflated_walls = obs.detect_dynamic_obstacles(cropped_image, robot_corners=robot_corners)

        # overlay obstacles onto annotated image
        annotated_img = obsv.draw_obstacles(annotated_img, raw_walls, inflated_walls)

        if robot is not None and "green" in drop_zones:
            start_pos = robot["center"]
            target_pos = drop_zones["green"]

            # 1.find path using a* on the inflated obstacle mask
            raw_path = pf.astar_search(inflated_walls, start_pos, target_pos, step_size=15)
            waypoints = pf.prune_waypoints(raw_path, min_dist=40.0)

            # 2.draw path lines and waypoints
            if waypoints:
                for i in range(len(waypoints) - 1):
                    cv2.line(annotated_img, waypoints[i], waypoints[i+1], (0, 255, 0), 3)
                for pt in waypoints:
                    cv2.circle(annotated_img, pt, 5, (255, 0, 0), -1)

        cv2.imshow("A* Navigation Result", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    
# ==================================