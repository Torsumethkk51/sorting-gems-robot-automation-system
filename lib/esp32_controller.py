# ========== ESP32 CONTROLLER & CLOSED-LOOP LOGIC ==========

import socket
import numpy as np

class ESP32Client:
    def __init__(self, esp_ip: str = "10.34.160.227", esp_port: int = 8888):
        self.target = (esp_ip, esp_port)
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

    def send_drive(self, left_pwm: int, right_pwm: int):
        # แปลงค่าความเร็วล้อซ้าย-ขวา เป็นตัวอักษรเดี่ยว
        if left_pwm == 0 and right_pwm == 0:
            self.send_cmd("S")
        elif left_pwm < 0 and right_pwm > 0:
            self.send_cmd("L")
        elif left_pwm > 0 and right_pwm < 0:
            self.send_cmd("R")
        elif left_pwm > 0 and right_pwm > 0:
            self.send_cmd("F")

    def send_action(self, action: str):
        # แปลงท่า GRAB และ DROP เป็น G และ D
        if action == "GRAB":
            self.send_cmd("G")
        elif action == "DROP":
            self.send_cmd("D")

    def stop(self):
        self.send_cmd("S")

    def send_action(self, action: str):
        # format action of robot
        msg = f"A,{action}\n".encode("utf-8")
        try:
            self.sock.sendto(msg, self.target)
        except OSError:
            pass

    def stop(self):
        self.send_drive(0, 0)

def track_waypoint_controller(robot_pos: tuple, robot_theta: float, target_pt: tuple, base_speed: int = 140, kp: float = 85.0):
    # distance betwenn robot and target
    dx = target_pt[0] - robot_pos[0]
    dy = target_pt[1] - robot_pos[1]
    dist = np.hypot(dx, dy)

    # arrived destination
    if dist < 20.0:
        return 0, 0, True

    target_heading = np.arctan2(dy, dx)
    diff = (target_heading - robot_theta + np.pi) % (2 * np.pi) - np.pi

    # turn in place if diff is more than 45rad
    if abs(diff) > np.radians(45):
        spin = 120
        return (-spin, spin, False) if diff > 0 else (spin, -spin, False)

    # move forward and turn in the same time
    turn = diff * kp
    left_pwm = int(base_speed - turn)
    right_pwm = int(base_speed + turn)
    return left_pwm, right_pwm, False

# ==========================================================
