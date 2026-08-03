"""3개 포즈 순환 테스트.

실행 전 확인:
  - GUI 닫혀 있어야 함 (시리얼 포트 점유 충돌)
  - USB 연결됨

실행:
  source ~/aero-hand/.venv/bin/activate
  python ~/aero-hand/test_poses.py
"""

import time
from aero_open_sdk.aero_hand import AeroHand

from poses import POSES, send_normalized

PORT = "/dev/cu.usbmodem101"
HOLD_SECONDS = 2.0  # 각 포즈 유지 시간

hand = AeroHand(port=PORT)

try:
    for name in ["paper", "rock", "scissors", "paper"]:
        print(f"→ {name}")
        send_normalized(hand, POSES[name])
        time.sleep(HOLD_SECONDS)
finally:
    hand.close()
    print("done")
