import socket
import time

ESP_IP = "10.34.160.208"
PORT = 4210

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"Connecting to ESP32 at {ESP_IP}:{PORT}...")

# ยิงตรงแบบ Unicast 5 ครั้ง
for i in range(1, 6):
    msg = f"F"
    sock.sendto(msg.encode("utf-8"), (ESP_IP, PORT))
    print(f"[Sent] -> {msg.strip()}")
    time.sleep(0.8)

print("Finished sending. ถามเพื่อนได้เลยค่ะว่าข้อความขึ้นบนหน้าจอไหม!")