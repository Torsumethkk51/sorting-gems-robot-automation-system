# ========== TEST CLOSED-LOOP NAVIGATION ==========

import cv2
import time
from lib import aruco_tracking as at
from lib.esp32_controller import ESP32Client, track_waypoint_controller

# 1. กำหนด IP ของ ESP32 (แก้ให้ตรงกับบอร์ดหน้างาน)
ESP32_IP = "192.168.4.1"
ESP32_PORT = 8888

# 2. รายการจุดพิกัด Waypoints จำลองสำหรับทดสอบ (Pixel coordinates)
# ตัวอย่าง: วิ่งเป็นสี่เหลี่ยม หรือกำหนดจุดทดสอบจุดแรก
TEST_WAYPOINTS = [
    (300, 300),
    (500, 300),
    (500, 500),
    (300, 500)
]

def run_navigation(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    esp = ESP32Client(esp_ip=ESP32_IP, esp_port=ESP32_PORT)
    waypoint_idx = 0
    total_waypoints = len(TEST_WAYPOINTS)

    print(f"[INFO] Connected to ESP32 at {ESP32_IP}:{ESP32_PORT}")
    print("[INFO] Starting navigation. Press 'q' to stop.")

    try:
        while waypoint_idx < total_waypoints:
            ret, frame = cap.read()
            if not ret:
                break

            target_pt = TEST_WAYPOINTS[waypoint_idx]

            # ตรวจจับตำแหน่งและมุมหันของหุ่นยนต์จาก ArUco
            robot = at.aruco_tracker(frame)

            if robot is not None:
                robot_pos = robot["center"]
                robot_theta = robot["theta_rad"]

                # คำนวณความเร็วล้อซ้าย-ขวาด้วย P-Controller
                lp, rp, reached = track_waypoint_controller(
                    robot_pos=robot_pos,
                    robot_theta=robot_theta,
                    target_pt=target_pt,
                    base_speed=130,   # ปรับความเร็วพื้นฐานเริ่มต้น
                    kp=80.0          # ปรับเกนการเลี้ยว
                )

                if reached:
                    print(f"[REACHED] Waypoint {waypoint_idx + 1}/{total_waypoints}: {target_pt}")
                    esp.stop()
                    waypoint_idx += 1
                    time.sleep(0.3)  # หยุดพักสั้นๆ ก่อนไปจุดถัดไป
                else:
                    esp.send_drive(lp, rp)

                # วาด Visual Feedback บนจอ
                cv2.circle(frame, robot_pos, 8, (0, 255, 0), -1)
                cv2.circle(frame, target_pt, 12, (0, 0, 255), 2)
                cv2.line(frame, robot_pos, target_pt, (255, 255, 0), 2)
                cv2.putText(frame, f"Target WP: {target_pt} | LP: {lp} RP: {rp}", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            else:
                # กรณีหลุดเฟรมหรือไม่เจอมาร์กเกอร์ ให้หยุดล้อเพื่อความปลอดภัย
                esp.stop()
                cv2.putText(frame, "[WARNING] ArUco Lost - Motors Stopped", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Real-Time Closed-Loop Navigation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # ปิดการทำงานและหยุดมอเตอร์ทุกกรณี
        esp.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Navigation stopped safely.")

# =================================================