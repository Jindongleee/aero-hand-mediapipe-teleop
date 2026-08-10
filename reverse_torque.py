#!/usr/bin/env python3
"""Aero Hand - 역방향 토크 제어 스크립트
사용법: python reverse_torque.py [토크값 0~1000] [서보ID or 'all']

예시:
  python reverse_torque.py 200 all     # 전체 서보 역방향 토크 200
  python reverse_torque.py 300 3       # 3번(index) 역방향 토크 300
  python reverse_torque.py 0 all       # 전체 토크 해제
"""
import sys
import struct
from serial import Serial

from config import detect_port, BAUD

CTRL_TOR = 0x12

def send_raw(ser, header, payload):
    msg = struct.pack("<2B7H", header & 0xFF, 0x00, *(v & 0xFFFF for v in payload))
    ser.write(msg)
    ser.flush()

def main():
    torque_val = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    target = sys.argv[2] if len(sys.argv) > 2 else "all"

    if torque_val == 0:
        raw_val = 0
    else:
        # 역방향: 비트10(1024) + 토크값
        raw_val = 1024 + min(torque_val, 1000)

    if target == "all":
        payload = [raw_val] * 7
    else:
        servo_id = int(target)
        payload = [0] * 7
        payload[servo_id] = raw_val

    ser = Serial(detect_port(), BAUD, timeout=0.01, write_timeout=0.01)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    send_raw(ser, CTRL_TOR, payload)
    direction = "역방향" if torque_val > 0 else "해제"
    print(f"토크 {direction}: 값={torque_val}, 대상={target}, raw={raw_val}")

    ser.close()

if __name__ == "__main__":
    main()
