# ========== MAIN LIVE AUTONOMOUS MISSION ==========

import cv2
import time
import copy
import numpy as np

from lib import vision_cleansing as vc
from lib import aruco_tracking as at
from lib import color_filtering as cf
from lib import object_detector as od
from lib import obstacles_detector as obs
from lib import obstacles_visualization as obsv
from lib import gem_selector as gs
from lib.esp32_controller import ESP32Client, track_waypoint_controller

# --- CONFIGURATION ---
ESP32_IP = "Your IP"
ESP32_PORT = 4210
CAMERA_INDEX = 0
BASE_SPEED = 135
KP_GAIN = 85.0

def execute_path_closed_loop(cap, esp: ESP32Client, path: list, action_at_end: str = None):
    """
    คุมหุ่นยนต์วิ่งตามชุด Waypoint ทีละจุดแบบ Closed-loop
    """
    if not path:
        return None

    for target_pt in path:
        while True:
            ret, frame = cap.read()
            if not ret:
                esp.stop()
                return None

            robot = at.aruco_tracker(frame)
            if robot is not None:
                robot_pos = robot["center"]
                robot_theta = robot["theta_rad"]

                lp, rp, reached = track_waypoint_controller(
                    robot_pos=robot_pos,
                    robot_theta=robot_theta,
                    target_pt=target_pt,
                    base_speed=BASE_SPEED,
                    kp=KP_GAIN
                )

                if reached:
                    break
                else:
                    esp.send_drive(lp, rp)

                # Feedback visual
                cv2.circle(frame, robot_pos, 7, (0, 255, 0), -1)
                cv2.circle(frame, target_pt, 8, (0, 0, 255), 2)
            else:
                esp.stop()

            cv2.imshow("Mission Live", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                esp.stop()
                return None

    # เมื่อเดินครบเส้นทาง ให้ทำ Action ที่กำหนด (เช่น GRAB หรือ DROP)
    esp.stop()
    if action_at_end:
        print(f"[ACTION] Triggering: {action_at_end}")
        esp.send_action(action_at_end)
        time.sleep(1.0)  # หน่วงเวลารอกลไกเซอร์โวทำงานเสร็จสิ้น

    return robot_pos if robot is not None else None

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot access camera.")
        return

    esp = ESP32Client(esp_ip=ESP32_IP, esp_port=ESP32_PORT)
    print(f"[INFO] UDP connected to ESP32: {ESP32_IP}:{ESP32_PORT}")

    # 1. ถ่ายภาพตั้งต้น 1 เฟรมเพื่อแมป Drop Zone, Gems และสิ่งกีดขวาง
    ret, initial_frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame from camera.")
        return

    print("[INFO] Processing initial field layout...")
    robot = at.aruco_tracker(initial_frame)
    current_color_range = cf.color_filtering(initial_frame)
    annotated_img, drop_zones, gems = od.detect_objects(initial_frame, current_color_range)

    robot_corners = robot["corners"] if robot is not None else None
    raw_walls, inflated_walls = obs.detect_dynamic_obstacles(initial_frame, robot_corners=robot_corners)

    active_gems = copy.deepcopy(gems)
    current_robot_pos = robot["center"] if robot is not None else (100, 100)
    step_counter = 1

    try:
        while True:
            total_remaining = sum(len(lst) for lst in active_gems.values())
            if total_remaining == 0:
                print("\n[ALL GEMS COLLECTED] Mission completed successfully!")
                break

            # 2. วางแผนเส้นทาง A*
            best_gem, pickup_path, drop_path, total_cost = gs.find_best_mission(
                robot_pos=current_robot_pos,
                gems_dict=active_gems,
                drop_zones=drop_zones,
                static_walls=raw_walls,
                step_size=10
            )

            if best_gem is None:
                print("\n[BLOCKED] No valid path to remaining gems!")
                break

            target_color = best_gem["color"]
            target_pos = best_gem["pos"]

            print(f"\n--- STEP {step_counter} ---")
            print(f"Target: {target_color.upper()} at {target_pos} | Remaining: {total_remaining}")

            # 3. เดินขาไปเก็บหิน (Pickup Leg) แล้วสั่ง GRAB
            last_pos = execute_path_closed_loop(cap, esp, pickup_path, action_at_end="GRAB")
            if last_pos is None:
                break

            # 4. เดินขานำไปส่ง (Drop Leg) แล้วสั่ง DROP
            last_pos = execute_path_closed_loop(cap, esp, drop_path, action_at_end="DROP")
            if last_pos is None:
                break

            # ปลดหินที่เก็บแล้วออกจากรายการ และอัปเดตตำแหน่งหุ่น
            active_gems[target_color].remove(target_pos)
            current_robot_pos = last_pos
            step_counter += 1

    finally:
        esp.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] System shutdown safely.")

if __name__ == "__main__":
    main()