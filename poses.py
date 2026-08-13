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

import config

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


def send_normalized(hand, normalized: list[float], side: str | None = None) -> None:
    """슬라이더 정규화 값 (0~1) 7개를 actuation degree로 변환해 손에 전송.

    GUI 슬라이더와 동일한 매핑:
      slider 0.0 → actuation_lower_limit
      slider 1.0 → actuation_upper_limit

    엄지(ch0 벌림 / ch1 굽힘)는 개체·펌웨어(좌/우)마다 편안한 방향이 반대이고,
    반대쪽 끝은 하드스톱에 눌려 서보가 수 A 를 뽑으며 탄다. 그래서 연결된 손을
    MAC 으로 판별해 `config.THUMB_SAFE_RANGE` 로 ch0·ch1 을 **재매핑**한다
    (정규화 [0,1] → 손별 안전 [lo,hi]). 손이 한 대뿐이라 자세 정확도보다
    서보 보호가 우선. 미측정 손은 레거시 CH0_MIN 하한만 건다.
    """
    if side is None:
        side = config.detect_hand_side()

    clamped = list(normalized)
    rng = config.THUMB_SAFE_RANGE.get(side)
    if rng is not None:
        ch0_lo, ch0_hi, ch1_lo, ch1_hi = rng
        clamped[0] = ch0_lo + clamped[0] * (ch0_hi - ch0_lo)
        clamped[1] = ch1_lo + clamped[1] * (ch1_hi - ch1_lo)
    else:
        clamped[0] = max(CH0_MIN, clamped[0])   # 레거시 (미측정 손)

    actuations = [
        hand.actuation_lower_limits[i]
        + n * (hand.actuation_upper_limits[i] - hand.actuation_lower_limits[i])
        for i, n in enumerate(clamped)
    ]
    hand.set_actuations(actuations)
