# ========== MAIN PROGRAM ==========

import copy
import cv2
from lib import vision_cleansing as vc
from lib import aruco_tracking as at
from lib import color_filtering as cf
from lib import object_detector as od
from lib import obstacles_detector as obs
from lib import obstacles_visualization as obsv
from lib import gem_selector as gs

if __name__ == "__main__":
    image = cv2.imread("test-image/field.png")
    test_arUco_image = cv2.imread("test-image/field_aruco_1.png")
    field_with_drop_zone = cv2.imread("test-image/field_with_drop_zone_obstacle_aruco2.png")

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

        # make a deep copy so we can pop collected gems iteratively
        active_gems = copy.deepcopy(gems)
        current_robot_pos = robot["center"] if robot is not None else (100, 100)

        step_counter = 1

        while True:
            # calculate total remaining gems
            total_remaining = sum(len(lst) for lst in active_gems.values())
            if total_remaining == 0:
                print("\n[ALL GEMS COLLECTED] mission completed successfully!")
                break

            # 1.find optimal target gem and paths
            best_gem, pickup_path, drop_path, total_cost = gs.find_best_mission(
                robot_pos=current_robot_pos,
                gems_dict=active_gems,
                drop_zones=drop_zones,
                static_walls=raw_walls,
                step_size=10  # use 10 for fine grid resolution
            )

            if best_gem is None:
                print("\n[BLOCKED] cannot find path to any remaining gems!")
                break

            target_color = best_gem["color"]
            target_pos = best_gem["pos"]
            drop_pos = drop_zones[target_color]

            # 2.print telemetry
            print(f"\n--- STEP {step_counter} ---")
            print(f"Robot Start : {current_robot_pos}")
            print(f"Target Gem  : {target_color.upper()} at {target_pos}")
            print(f"Drop Zone   : {target_color.upper()} at {drop_pos}")
            print(f"Total Dist  : {total_cost:.1f} px")
            print(f"Remaining   : {total_remaining} gems")

            # 3.render mission overview path
            mission_frame = gs.render_mission(
                image=annotated_img,
                robot_pos=current_robot_pos,
                best_gem=best_gem,
                drop_pos=drop_pos,
                pickup_path=pickup_path,
                drop_path=drop_path,
                remaining_count=total_remaining
            )

            # 4.run real-time simulation motion
            # combine pickup and drop paths seamlessly
            full_route = pickup_path + drop_path[1:] if drop_path else pickup_path
            
            # robot physically animates along path
            end_robot_pos = gs.simulate_robot_motion(
                base_frame=mission_frame,
                full_path=full_route,
                speed_px=10.0,   # adjust speed of animation
                delay_ms=10      # frame rate delay
            )

            # remove collected gem and update robot start pos to actual stopped location
            active_gems[target_color].remove(target_pos)
            current_robot_pos = end_robot_pos
            step_counter += 1

        cv2.destroyAllWindows()
        
    
# ==================================