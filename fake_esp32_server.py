# ========== FAKE ESP32 UDP SERVER ==========

import socket

SERVER_IP = "127.0.0.1"  # Localhost บนเครื่องตัวเอง
SERVER_PORT = 8888

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((SERVER_IP, SERVER_PORT))

print(f"[FAKE ESP32] Listening for UDP packets on {SERVER_IP}:{SERVER_PORT}")
print("[FAKE ESP32] Ready to simulate motor & servo responses...\n")

try:
    while True:
        data, addr = sock.recvfrom(1024)
        msg = data.decode("utf-8").strip()

        if msg.startswith("V,"):
            # ขับเคลื่อนมอเตอร์: V,left_pwm,right_pwm
            parts = msg.split(",")
            lp, rp = parts[1], parts[2]
            print(f"[DRIVE] Left PWM: {lp:>4} | Right PWM: {rp:>4}")

        elif msg.startswith("A,"):
            # แอ็กชันแขนกล: A,GRAB หรือ A,DROP
            action = msg.split(",")[1]
            print(f">>> [SERVO ACTION] Executing: {action} <<<")

except KeyboardInterrupt:
    print("\n[FAKE ESP32] Shutting down server.")
finally:
    sock.close()

# ===========================================