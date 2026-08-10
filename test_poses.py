"""3개 포즈 순환 테스트.

포즈 사이를 한 번에 점프하지 않고 보간해서 이동한다. 텐던 구동 손이라
급격한 위치 변화는 줄에 충격을 준다. 손이 한 대뿐이라 느린 쪽을 택한다.

실행:
  .venv/bin/python test_poses.py

포트는 SDK 가 자동 탐지한다 (config.py 참고).
"""

import struct
import time

from poses import POSES, PAPER, send_normalized
from config import open_hand, N_CHANNELS

HOLD_SECONDS = 2.5    # 각 포즈 유지 시간
MOVE_SECONDS = 2.0    # 포즈 사이 이동 시간
RATE_HZ = 30

SEQUENCE = ["paper", "rock", "scissors", "paper"]


def move_to(hand, start, goal, seconds=MOVE_SECONDS):
    steps = max(1, int(seconds * RATE_HZ))
    for i in range(1, steps + 1):
        t = i / steps
        send_normalized(hand, [s + (g - s) * t for s, g in zip(start, goal)])
        time.sleep(1.0 / RATE_HZ)


def main():
    hand = open_hand()
    print(f"connected: {hand.ser.port}")

    current = list(PAPER)
    try:
        # 현재 자세를 모르므로 PAPER 로 천천히 모은 뒤 시작한다.
        send_normalized(hand, current)
        time.sleep(1.0)

        for name in SEQUENCE:
            print(f"→ {name}")
            goal = list(POSES[name])
            move_to(hand, current, goal)
            current = goal
            time.sleep(HOLD_SECONDS)
    finally:
        msg = struct.pack("<2B7H", 0x12, 0x00, *[0] * N_CHANNELS)
        hand.ser.write(msg)
        hand.ser.flush()
        hand.close()
        print("done (torque released)")


if __name__ == "__main__":
    main()
