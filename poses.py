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

# ch0(엄지 벌림)은 0.05 미만에서 기계적 한계에 눌린다.
# 실측: 0.02 를 유지하면 모터가 -110 mA 를 계속 뽑으며 56도까지 올랐고,
# 0.05 이상에서는 2~14 mA 로 떨어진다. 그래서 하한을 CH0_MIN 으로 둔다.
CH0_MIN = 0.06

ROCK     = [0.069, 0.489, 0.831, 0.931, 0.980, 0.978, 0.979]
SCISSORS = [0.617, 0.843, 0.863, 0.000, 0.000, 0.978, 0.979]
PAPER    = [CH0_MIN, 0.000, 0.058, 0.000, 0.000, 0.000, 0.000]

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

    ch0 만 CH0_MIN 으로 하한을 건다. 미러링 중에는 캘리브레이션이 0~1 전체를
    쓰기 때문에 ch0 이 수시로 0 까지 내려가는데, 그 구간에서는 엄지 벌림이
    기계적 한계에 눌려 모터가 계속 전류를 뽑는다 (실측 -110 mA, 56도).
    손이 한 대뿐이라 서보를 태우는 쪽이 자세 정확도보다 비싸다.
    """
    clamped = list(normalized)
    clamped[0] = max(CH0_MIN, clamped[0])

    actuations = [
        hand.actuation_lower_limits[i]
        + n * (hand.actuation_upper_limits[i] - hand.actuation_lower_limits[i])
        for i, n in enumerate(clamped)
    ]
    hand.set_actuations(actuations)
