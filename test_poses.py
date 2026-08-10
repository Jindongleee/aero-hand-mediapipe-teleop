"""3개 포즈 순환 테스트.

실행 전 확인:
  - GUI 닫혀 있어야 함 (시리얼 포트 점유 충돌)
  - USB 연결됨

실행:
  .venv/bin/python test_poses.py

포트는 SDK 가 자동 탐지한다 (config.py 참고).
"""

import time

from poses import POSES, send_normalized
from config import open_hand

HOLD_SECONDS = 2.0  # 각 포즈 유지 시간

hand = open_hand()

try:
    for name in ["paper", "rock", "scissors", "paper"]:
        print(f"→ {name}")
        send_normalized(hand, POSES[name])
        time.sleep(HOLD_SECONDS)
finally:
    hand.close()
    print("done")
