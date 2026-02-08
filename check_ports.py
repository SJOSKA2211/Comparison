import socket
import sys

def check_port(port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('127.0.0.1', port))
            return result == 0
    except:
        return False

ports = {
    "Backend (8000)": 8000,
    "Frontend (3000)": 3000
}

print("Service Status check:")
for name, port in ports.items():
    status = "UP" if check_port(port) else "DOWN"
    print(f"{name}: {status}")

