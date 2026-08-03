"""가위바위보 포즈 정의 — Aero Hand

각 포즈는 슬라이더 정규화 값 0.0 ~ 1.0 의 7개 리스트.
채널 매핑:
  0 = 엄지 벌림 (thumb abduction)
  1 = 엄지 굽힘 (thumb flexion)
  2 = 엄지 텐던 (thumb tendon)
  3 = 검지 (index)
  4 = 중지 (middle)
  5 = 약지 (ring)
  6 = 소지 (pinky)
"""

ROCK     = [0.069, 0.489, 0.831, 0.931, 0.980, 0.978, 0.979]
SCISSORS = [0.617, 0.843, 0.863, 0.000, 0.000, 0.978, 0.979]
PAPER    = [0.018, 0.000, 0.058, 0.000, 0.000, 0.000, 0.000]

POSES = {
    "rock":     ROCK,
    "scissors": SCISSORS,
    "paper":    PAPER,
}


def send_normalized(hand, normalized: list[float]) -> None:
    """슬라이더 정규화 값 (0~1) 7개를 actuation degree로 변환해 손에 전송.

    GUI 슬라이더와 동일한 매핑:
      slider 0.0 → actuation_lower_limit
      slider 1.0 → actuation_upper_limit
    """
    actuations = [
        hand.actuation_lower_limits[i]
        + n * (hand.actuation_upper_limits[i] - hand.actuation_lower_limits[i])
        for i, n in enumerate(normalized)
    ]
    hand.set_actuations(actuations)
