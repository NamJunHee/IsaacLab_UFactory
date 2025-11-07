import socket
import struct
import torch
import time

# ========== CONFIG ==========
SOCKET_PATH = "/tmp/object.sock"
RECV_SIZE = 12  # 3 floats (x, y, z)
TIMEOUT_SEC = 1.0  # 1초 안에 수신 없으면 skip

# ========== FUNCTIONS ==========

def recv_exact(sock, n_bytes):
    """n_bytes 만큼 정확히 수신"""
    buf = b''
    while len(buf) < n_bytes:
        try:
            chunk = sock.recv(n_bytes - len(buf))
            if not chunk:
                return None
            buf += chunk
        except socket.timeout:
            return None
    return buf

# ========== SOCKET INIT ==========

client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
print("[Receiver] Connecting to YOLO socket...")
client.connect(SOCKET_PATH)
client.settimeout(TIMEOUT_SEC)
print("[Receiver] Connected! Listening for (x, y, z) data...")

# ========== MAIN LOOP ==========

while True:
    data = recv_exact(client, RECV_SIZE)
    if data and len(data) == 12:
        x, y, z = struct.unpack('fff', data)
        yolo_pos = torch.tensor([x, y, z])
        print(f"[Receiver] Received: x={x:.3f}, y={y:.3f}, z={z:.3f}")
    else:
        print("[Receiver] Timeout or invalid data.")
    
    time.sleep(0.01)  # Optional: 너무 자주 출력되면 조금 쉬게 함
