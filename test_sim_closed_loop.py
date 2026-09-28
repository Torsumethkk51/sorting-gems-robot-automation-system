# ========== SIMULATION WITH DISCRETE CHAR UDP ==========

import copy
import math
import socket
import time
import cv2
import numpy as np

from lib import vision_cleansing as vc
from lib import aruco_tracking as at
from lib import color_filtering as cf
from lib import object_detector as od
from lib import obstacles_detector as obs
from lib import gem_selector as gs

# ตั้งค่าการเชื่อมต่อ ESP32
ESP32_IP = "10.34.160.208"  # ใส่ IP ของบอร์ด ESP32
ESP32_PORT = 4210           # พอร์ต UDP ของ ESP32

class DiscreteESP32Client:
    def __init__(self, ip: str, port: int):
        self.target = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.last_cmd = ""

    def send_cmd(self, cmd: str):
        if cmd != self.last_cmd:
            try:
                self.sock.sendto(cmd.encode("utf-8"), self.target)
                self.last_cmd = cmd
                print(f"[UDP SENT] -> {cmd}")
            except OSError:
                pass

    def stop(self):
        self.send_cmd("S")


def simulate_discrete_movement(canvas, esp, path, current_pos, current_theta, action_at_end=None):
    pos = np.array(current_pos, dtype=np.float32)
    theta = float(current_theta)

    # ปรับสปีดจำลองและเวลารอให้เร็วขึ้นตามความเร็วจริง
    linear_speed = 130.0   # ความเร็วเดินหน้า (px/s)
    turn_speed = 3.5       # ความเร็วเลี้ยว (rad/s)

    for target_pt in path:
        dx = target_pt[0] - pos[0]
        dy = target_pt[1] - pos[1]
        dist = math.hypot(dx, dy)

        if dist < 10.0:
            continue

        # --------------------------------------------------
        # สเต็ป 1: หมุนตัวให้ตรงทิศก่อน (Turn Phase)
        # --------------------------------------------------
        target_angle = math.atan2(dy, dx)
        diff_angle = (target_angle - theta + np.pi) % (2 * np.pi) - np.pi

        if abs(diff_angle) > math.radians(18):
            turn_cmd = "R" if diff_angle > 0 else "L"
            esp.send_cmd(turn_cmd)

            turn_time = abs(diff_angle) / turn_speed
            start_t = time.time()
            while time.time() - start_t < turn_time:
                dt = 0.03
                theta += (turn_speed if diff_angle > 0 else -turn_speed) * dt
                
                frame = gs.draw_simulated_aruco(canvas, (pos[0], pos[1]), theta)
                cv2.circle(frame, target_pt, 5, (0, 0, 255), -1)
                cv2.putText(frame, f"CMD: {turn_cmd} (TURNING)", (25, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.imshow("Closed-Loop UDP Simulation", frame)
                if (cv2.waitKey(20) & 0xFF) == ord('q'):
                    esp.stop()
                    return (int(pos[0]), int(pos[1])), theta

            theta = target_angle
            esp.stop()
            time.sleep(0.12)  # พักหัวรถให้หยุดนิ่งสนิท

        # --------------------------------------------------
        # สเต็ป 2: เดินหน้าตรงรวดเดียวไปถึงจุด (Drive Phase)
        # --------------------------------------------------
        esp.send_cmd("F")
        drive_time = dist / linear_speed
        start_t = time.time()

        while time.time() - start_t < drive_time:
            dt = 0.03
            pos[0] += linear_speed * math.cos(theta) * dt
            pos[1] += linear_speed * math.sin(theta) * dt

            frame = gs.draw_simulated_aruco(canvas, (pos[0], pos[1]), theta)
            cv2.circle(frame, target_pt, 5, (0, 0, 255), -1)
            cv2.putText(frame, "CMD: F (DRIVING)", (25, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(frame, f"Dist: {dist:.1f} px", (25, 175), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imshow("Closed-Loop UDP Simulation", frame)
            if (cv2.waitKey(20) & 0xFF) == ord('q'):
                esp.stop()
                return (int(pos[0]), int(pos[1])), theta

        # ถึงจุด Waypoint นี้แล้ว หยุดก่อนไปจุดต่อไป
        pos[0], pos[1] = target_pt[0], target_pt[1]
        esp.stop()
        time.sleep(0.15)

    # --------------------------------------------------
    # สเต็ป 3: ถึงเป้าหมายปลายทาง สั่งคีบหรือปล่อย
    # --------------------------------------------------
    if action_at_end == "GRAB":
        esp.send_cmd("G")
        time.sleep(1.2)
    elif action_at_end == "DROP":
        esp.send_cmd("D")
        time.sleep(1.2)

    return (int(pos[0]), int(pos[1])), theta


if __name__ == "__main__":
    esp = DiscreteESP32Client(ip=ESP32_IP, port=ESP32_PORT)
    field_img = cv2.imread("test-image/field_with_drop_zone_obstacle_aruco2.png")

    if field_img is not None:
        cropped = vc.vision_cleansing(image=field_img)
        robot = at.aruco_tracker(image=cropped)
        color_range = cf.color_filtering(image=cropped)
        annotated_img, drop_zones, gems = od.detect_objects(cropped, color_range)

        robot_corners = robot["corners"] if robot is not None else None
        raw_walls, _ = obs.detect_dynamic_obstacles(cropped, robot_corners=robot_corners)

        active_gems = copy.deepcopy(gems)
        curr_pos = robot["center"] if robot is not None else (100, 100)
        curr_theta = robot["theta_rad"] if robot is not None else 0.0

        while True:
            if sum(len(v) for v in active_gems.values()) == 0:
                print("\n[FINISHED] All gems collected successfully!")
                break

            best_gem, pickup_path, drop_path, _ = gs.find_best_mission(
                robot_pos=curr_pos,
                gems_dict=active_gems,
                drop_zones=drop_zones,
                static_walls=raw_walls,
                step_size=10
            )

            if best_gem is None:
                break

            target_color = best_gem["color"]
            target_pos = best_gem["pos"]
            drop_pos = drop_zones[target_color]

            mission_frame = gs.render_mission(
                annotated_img, curr_pos, best_gem, drop_pos,
                pickup_path, drop_path, sum(len(v) for v in active_gems.values())
            )

            # ขาไปเก็บหิน -> ส่งคำสั่ง G ตอนถึง
            curr_pos, curr_theta = simulate_discrete_movement(
                mission_frame, esp, pickup_path, curr_pos, curr_theta, action_at_end="GRAB"
            )

            # ขานำไปส่ง Drop Zone -> ส่งคำสั่ง D ตอนถึง
            curr_pos, curr_theta = simulate_discrete_movement(
                mission_frame, esp, drop_path, curr_pos, curr_theta, action_at_end="DROP"
            )

            active_gems[target_color].remove(target_pos)

        esp.stop()
        cv2.destroyAllWindows()